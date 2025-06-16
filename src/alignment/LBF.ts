import { LBFRegressor } from "./Regressor";
import { Image } from "../utils/Image";
import { Matrix, MatrixType } from "../math/Matrix";
import { RegressorData } from "./training/Regressor";

/**
 * Face Alignment via Regressing Local Binary Features (LBF)
 * This approach has two components: a set of local binary features and
 * a locality principle for learning those features.
 * The locality principle is used to guide the learning of a set of highly
 * discriminative local binary features for each landmark independently.
 * The obtained local binary features are used to learn a linear regression
 * that later will be used to guide the landmarks in the alignment phase.
 *
 * @authors: VoxarLabs Team (http://cin.ufpe.br/~voxarlabs)
 *           Lucas Figueiredo <lsf@cin.ufpe.br>, Thiago Menezes <tmc2@cin.ufpe.br>,
 *           Thiago Domingues <tald@cin.ufpe.br>, Rafael Roberto <rar3@cin.ufpe.br>,
 *           Thulio Araujo <tlsa@cin.ufpe.br>, Joao Victor <jvfl@cin.ufpe.br>,
 *           Tomer Simis <tls@cin.ufpe.br>)
 */
interface BoundingBox {
    startX: number;
    startY: number;
    width: number;
    height: number;
}

interface Face {
    x: number;
    y: number;
    width: number;
    height: number;
}

export class LBF {
    /**
     * Holds the maximum number of stages that will be used in the alignment algorithm.
     * Each stage contains a different set of random forests and retrieves the binary
     * code from a more "specialized" (i.e. smaller) region around the landmarks.
     */
    static readonly maxNumStages: number = 4;

    /**
     * Holds the regressor that will be responsible for extracting the local features from
     * the image and guide the landmarks using the training data.
     */
    static regressor_: LBFRegressor | null = null;

    /**
     * Generates a set of landmarks for a set of faces
     * @param pixels The pixels in a linear [r,g,b,a,...] array.
     * @param width The image width.
     * @param height The image height.
     * @param faces The list of faces detected in the image
     * @return The aligned landmarks, each set of landmarks corresponding to a specific face.
     */
    public static align(pixels: Uint8ClampedArray, width: number, height: number, faces: Face[]): MatrixType[] {
        if (LBF.regressor_ === null) {
            LBF.regressor_ = new LBFRegressor(LBF.maxNumStages);
        }

        // NOTE: is this thresholding suitable ? if it is on image, why no skin-color filter ? and a adaptative threshold
        pixels = Image.equalizeHist(Image.grayscale(pixels, width, height, false), width, height);

        const shapes: MatrixType[] = new Array(faces.length);

        for (let i = 0; i < faces.length; i++) {
            faces[i].height = faces[i].width;

            const boundingBox: BoundingBox = {
                startX: faces[i].x,
                startY: faces[i].y,
                width: faces[i].width,
                height: faces[i].height,
            };

            shapes[i] = LBF.regressor_.predict(pixels, width, height, boundingBox);
        }

        return shapes;
    }

    /**
     * Unprojects the landmarks shape from the bounding box.
     * @param shape The landmarks shape.
     * @param boundingBox The bounding box.
     * @return The landmarks shape projected into the bounding box.
     */
    public static unprojectShapeToBoundingBox(shape: MatrixType, boundingBox: BoundingBox): MatrixType {
        const temp: MatrixType = new Array(shape.length);
        for (let i = 0; i < shape.length; i++) {
            temp[i] = [
                (shape[i][0] - boundingBox.startX) / boundingBox.width,
                (shape[i][1] - boundingBox.startY) / boundingBox.height,
            ];
        }
        return temp;
    }

    /**
     * Projects the landmarks shape into the bounding box. The landmarks shape has
     * normalized coordinates, so it is necessary to map these coordinates into
     * the bounding box coordinates.
     * @param shape The landmarks shape.
     * @param boundingBox The bounding box.
     * @return The landmarks shape.
     */
    public static projectShapeToBoundingBox(shape: MatrixType, boundingBox: BoundingBox): MatrixType {
        const temp: MatrixType = new Array(shape.length);
        for (let i = 0; i < shape.length; i++) {
            temp[i] = [
                shape[i][0] * boundingBox.width + boundingBox.startX,
                shape[i][1] * boundingBox.height + boundingBox.startY,
            ];
        }
        return temp;
    }

    /**
     * Calculates the rotation and scale necessary to transform shape1 into shape2.
     * @param shape1 The shape to be transformed.
     * @param shape2 The shape to be transformed in.
     * @return The rotation matrix and scale that applied to shape1 results in shape2.
     */
    public static similarityTransform(shape1: MatrixType, shape2: MatrixType): [MatrixType, number] {
        const center1: number[] = [0, 0];
        const center2: number[] = [0, 0];

        for (let i = 0; i < shape1.length; i++) {
            center1[0] += shape1[i][0];
            center1[1] += shape1[i][1];
            center2[0] += shape2[i][0];
            center2[1] += shape2[i][1];
        }

        center1[0] /= shape1.length;
        center1[1] /= shape1.length;
        center2[0] /= shape2.length;
        center2[1] /= shape2.length;

        let temp1: MatrixType = Matrix.clone(shape1);
        let temp2: MatrixType = Matrix.clone(shape2);

        for (let i = 0; i < shape1.length; i++) {
            temp1[i][0] -= center1[0];
            temp1[i][1] -= center1[1];
            temp2[i][0] -= center2[0];
            temp2[i][1] -= center2[1];
        }

        let t = Matrix.calcCovarMatrix(temp1);
        let covariance1: MatrixType = t[0];
        let mean1: MatrixType = t[1];

        t = Matrix.calcCovarMatrix(temp2);
        let covariance2: MatrixType = t[0];
        let mean2: MatrixType = t[1];

        const s1: number = Math.sqrt(Matrix.norm(covariance1));
        const s2: number = Math.sqrt(Matrix.norm(covariance2));

        const scale: number = s1 / s2;
        temp1 = Matrix.mulScalar(1.0 / s1, temp1);
        temp2 = Matrix.mulScalar(1.0 / s2, temp2);

        let num: number = 0;
        let den: number = 0;

        for (let i = 0; i < shape1.length; i++) {
            num = num + temp1[i][1] * temp2[i][0] - temp1[i][0] * temp2[i][1];
            den = den + temp1[i][0] * temp2[i][0] + temp1[i][1] * temp2[i][1];
        }

        const norm: number = Math.sqrt(num * num + den * den);
        const sin_theta: number = num / norm;
        const cos_theta: number = den / norm;
        const rotation: MatrixType = [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ];

        return [rotation, scale];
    }
}

/**
 * LBF Random Forest data structure.
 */
export class LBFRandomForest {
    public readonly maxNumTrees: number;
    public readonly landmarkNum: number;
    public readonly maxDepth: number;
    public readonly stages: any;
    public readonly rfs: LBFTree[][];

    constructor(forestIndex: number) {
        this.maxNumTrees = RegressorData[forestIndex].max_numtrees;
        this.landmarkNum = RegressorData[forestIndex].num_landmark;
        this.maxDepth = RegressorData[forestIndex].max_depth;
        this.stages = RegressorData[forestIndex].stages;

        this.rfs = new Array(this.landmarkNum);
        for (let i = 0; i < this.landmarkNum; i++) {
            this.rfs[i] = new Array(this.maxNumTrees);
            for (let j = 0; j < this.maxNumTrees; j++) {
                this.rfs[i][j] = new LBFTree(forestIndex, i, j);
            }
        }
    }
}

/**
 * LBF Tree data structure.
 */
export class LBFTree {
    public readonly maxDepth: number;
    public readonly maxNumNodes: number;
    public readonly nodes: any;
    public readonly landmarkID: number;
    public readonly numLeafnodes: number;
    public readonly numNodes: number;
    public readonly maxNumFeats: number;
    public readonly maxRadioRadius: number;
    public readonly leafnodes: any;

    constructor(forestIndex: number, landmarkIndex: number, treeIndex: number) {
        const data = RegressorData[forestIndex].landmarks[landmarkIndex][treeIndex];
        this.maxDepth = data.max_depth;
        this.maxNumNodes = data.max_numnodes;
        this.nodes = data.nodes;
        this.landmarkID = data.landmark_id;
        this.numLeafnodes = data.num_leafnodes;
        this.numNodes = data.num_nodes;
        this.maxNumFeats = data.max_numfeats;
        this.maxRadioRadius = data.max_radio_radius;
        this.leafnodes = data.id_leafnodes;
    }
}
