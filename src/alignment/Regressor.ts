import { Matrix } from "../math/Matrix";
import { LBF, LBFRandomForest } from "./LBF";
import { LandmarksData } from "./training/Landmarks";
import { RegressorData } from "./training/Regressor";

export class LBFRegressor {
    private readonly maxNumStages: number;
    private readonly rfs: any[];
    private readonly models: any[];
    private readonly meanShape: any;

    constructor(maxNumStages: number) {
        this.maxNumStages = maxNumStages;

        this.rfs = new Array(maxNumStages);
        this.models = new Array(maxNumStages);

        for (let i = 0; i < maxNumStages; i++) {
            this.rfs[i] = new LBFRandomForest(i);
            this.models[i] = RegressorData[i].models;
        }

        this.meanShape = LandmarksData;
    }

    /**
     * Predicts the position of the landmarks based on the bounding box of the face.
     * @param pixels The grayscale pixels in a linear array.
     * @param width Width of the image.
     * @param height Height of the image.
     * @param boundingBox Bounding box of the face to be aligned.
     * @return A matrix with each landmark position in a row [x,y].
     */
    predict(pixels: Uint8ClampedArray, width: number, height: number, boundingBox: any): any {
        const images: any[] = [];
        const currentShapes: any[] = [];
        const boundingBoxes: any[] = [];

        const meanShapeClone = Matrix.clone(this.meanShape);

        images.push({
            data: pixels,
            width: width,
            height: height,
        });
        boundingBoxes.push(boundingBox);

        currentShapes.push(LBF.projectShapeToBoundingBox(meanShapeClone, boundingBox));

        for (let stage = 0; stage < this.maxNumStages; stage++) {
            const binaryFeatures = LBFRegressor.deriveBinaryFeat(
                this.rfs[stage],
                images,
                currentShapes,
                boundingBoxes,
                meanShapeClone
            );
            this.applyGlobalPrediction(binaryFeatures, this.models[stage], currentShapes, boundingBoxes);
        }

        return currentShapes[0];
    }

    /**
     * Multiplies the binary features of the landmarks with the regression matrix
     * to obtain the displacement for each landmark. Then applies this displacement
     * into the landmarks shape.
     * @param binaryFeatures The binary features for the landmarks.
     * @param models The regressor models.
     * @param currentShapes The landmarks shapes.
     * @param boundingBoxes The bounding boxes of the faces.
     */
    applyGlobalPrediction(binaryFeatures: any, models: any, currentShapes: any[], boundingBoxes: any[]): void {
        const residual = currentShapes[0].length * 2;

        let rotation: any[] = [];
        const deltashape = new Array(residual / 2);
        for (let i = 0; i < residual / 2; i++) {
            deltashape[i] = [0.0, 0.0];
        }

        for (let i = 0; i < currentShapes.length; i++) {
            for (let j = 0; j < residual; j++) {
                let tmp = 0;
                for (let lx = 0, idx = 0; (idx = binaryFeatures[i][lx].index) != -1; lx++) {
                    if (idx <= models[j].nr_feature) {
                        tmp += models[j].data[idx - 1] * binaryFeatures[i][lx].value;
                    }
                }
                if (j < residual / 2) {
                    deltashape[j][0] = tmp;
                } else {
                    deltashape[j - residual / 2][1] = tmp;
                }
            }

            const res = LBF.similarityTransform(
                LBF.unprojectShapeToBoundingBox(currentShapes[i], boundingBoxes[i]),
                this.meanShape
            );
            rotation = Matrix.transpose(res[0]);

            let s = LBF.unprojectShapeToBoundingBox(currentShapes[i], boundingBoxes[i]);
            s = Matrix.add(s, deltashape);

            currentShapes[i] = LBF.projectShapeToBoundingBox(s, boundingBoxes[i]);
        }
    }

    /**
     * Derives the binary features from the image for each landmark.
     * @param forest The random forest to search for the best binary feature match.
     * @param images The images with pixels in a grayscale linear array.
     * @param currentShapes The current landmarks shape.
     * @param boundingBoxes The bounding boxes of the faces.
     * @param meanShape The mean shape of the current landmarks set.
     * @return The binary features extracted from the image and matched with the training data.
     */
    static deriveBinaryFeat(
        forest: any,
        images: any[],
        currentShapes: any[],
        boundingBoxes: any[],
        meanShape: any
    ): any[] {
        const binaryFeatures = new Array(images.length);
        for (let i = 0; i < images.length; i++) {
            const t = forest.maxNumTrees * forest.landmarkNum + 1;
            binaryFeatures[i] = new Array(t);
            for (let j = 0; j < t; j++) {
                binaryFeatures[i][j] = {};
            }
        }

        const leafnodesPerTree = 1 << (forest.maxDepth - 1);

        for (let i = 0; i < images.length; i++) {
            const projectedShape = LBF.unprojectShapeToBoundingBox(currentShapes[i], boundingBoxes[i]);
            const transform = LBF.similarityTransform(projectedShape, meanShape);

            for (let j = 0; j < forest.landmarkNum; j++) {
                for (let k = 0; k < forest.maxNumTrees; k++) {
                    const binaryCode = LBFRegressor.getCodeFromTree(
                        forest.rfs[j][k],
                        images[i],
                        currentShapes[i],
                        boundingBoxes[i],
                        transform[0],
                        transform[1]
                    );

                    const index = j * forest.maxNumTrees + k;
                    binaryFeatures[i][index].index = leafnodesPerTree * index + binaryCode;
                    binaryFeatures[i][index].value = 1;
                }
            }
            binaryFeatures[i][forest.landmarkNum * forest.maxNumTrees].index = -1;
            binaryFeatures[i][forest.landmarkNum * forest.maxNumTrees].value = -1;
        }
        return binaryFeatures;
    }

    /**
     * Gets the binary code for a specific tree in a random forest. For each landmark,
     * the position from two pre-defined points are recovered from the training data
     * and then the intensity of the pixels corresponding to these points is extracted
     * from the image and used to traverse the trees in the random forest. At the end,
     * the ending nodes will be represented by 1, and the remaining nodes by 0.
     * @param tree The tree to be analyzed.
     * @param image The image with pixels in a grayscale linear array.
     * @param shape The current landmarks shape.
     * @param boundingBox The bounding box of the face.
     * @param rotation The rotation matrix used to transform the projected landmarks into the mean shape.
     * @param scale The scale factor used to transform the projected landmarks into the mean shape.
     * @return The binary code extracted from the tree.
     */
    static getCodeFromTree(tree: any, image: any, shape: any, boundingBox: any, rotation: any, scale: number): number {
        let current = 0;
        let bincode = 0;

        while (true) {
            const x1 =
                Math.cos(tree.nodes[current].feats[0]) *
                tree.nodes[current].feats[2] *
                tree.maxRadioRadius *
                boundingBox.width;
            const y1 =
                Math.sin(tree.nodes[current].feats[0]) *
                tree.nodes[current].feats[2] *
                tree.maxRadioRadius *
                boundingBox.height;
            const x2 =
                Math.cos(tree.nodes[current].feats[1]) *
                tree.nodes[current].feats[3] *
                tree.maxRadioRadius *
                boundingBox.width;
            const y2 =
                Math.sin(tree.nodes[current].feats[1]) *
                tree.nodes[current].feats[3] *
                tree.maxRadioRadius *
                boundingBox.height;

            const project_x1 = rotation[0][0] * x1 + rotation[0][1] * y1;
            const project_y1 = rotation[1][0] * x1 + rotation[1][1] * y1;

            let real_x1 = Math.floor(project_x1 + shape[tree.landmarkID][0]);
            let real_y1 = Math.floor(project_y1 + shape[tree.landmarkID][1]);
            real_x1 = Math.max(0.0, Math.min(real_x1, image.height - 1.0));
            real_y1 = Math.max(0.0, Math.min(real_y1, image.width - 1.0));

            const project_x2 = rotation[0][0] * x2 + rotation[0][1] * y2;
            const project_y2 = rotation[1][0] * x2 + rotation[1][1] * y2;

            let real_x2 = Math.floor(project_x2 + shape[tree.landmarkID][0]);
            let real_y2 = Math.floor(project_y2 + shape[tree.landmarkID][1]);
            real_x2 = Math.max(0.0, Math.min(real_x2, image.height - 1.0));
            real_y2 = Math.max(0.0, Math.min(real_y2, image.width - 1.0));

            const pdf =
                Math.floor(image.data[real_y1 * image.width + real_x1]) -
                Math.floor(image.data[real_y2 * image.width + real_x2]);

            if (pdf < tree.nodes[current].thresh) {
                current = tree.nodes[current].cnodes[0];
            } else {
                current = tree.nodes[current].cnodes[1];
            }

            if (tree.nodes[current].is_leafnode == 1) {
                bincode = 1;
                for (let i = 0; i < tree.leafnodes.length; i++) {
                    if (tree.leafnodes[i] == current) {
                        return bincode;
                    }
                    bincode++;
                }
                return bincode;
            }
        }
    }
}
