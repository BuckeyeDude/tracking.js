import { Tracker } from "./Tracker";
import { ViolaJones } from "../detection/ViolaJones";
import { Rect } from "../detection/Rect";

/**
 * ObjectTracker utility class for tracking objects in video frames.
 * Extends the base Tracker class.
 */
export class ObjectTracker extends Tracker {
    /**
     * Specifies the edges density of a block in order to decide whether to skip
     * it or not.
     * @default 0.2
     */
    public edgesDensity: number = 0.2;

    /**
     * Specifies the initial scale to start the feature block scaling.
     * @default 1.0
     */
    public initialScale: number = 1.0;

    /**
     * Specifies the scale factor to scale the feature block.
     * @default 1.25
     */
    public scaleFactor: number = 1.25;

    /**
     * Specifies the block step size.
     * @default 1.5
     */
    public stepSize: number = 1.5;

    private classifiers?: Float64Array[];

    /**
     * Creates a new ObjectTracker instance.
     * @param optClassifiers Optional object classifiers to track.
     */
    constructor(optClassifiers?: string | string[]) {
        super();

        if (optClassifiers) {
            let classifiers = optClassifiers;
            let newClassifiers: Float64Array[] = [];

            if (!Array.isArray(classifiers)) {
                classifiers = [classifiers];
            }

            if (Array.isArray(classifiers)) {
                for (const classifier of classifiers) {
                    const i = classifiers.indexOf(classifier);
                    if (typeof classifier === "string") {
                        newClassifiers[i] = ViolaJones.classifiers[classifier];
                    }
                    if (!newClassifiers[i]) {
                        throw new Error('Object classifier not valid, try `new tracking.ObjectTracker("face")`.');
                    }
                }
            }

            this.setClassifiers(newClassifiers);
        }
    }

    /**
     * Gets the tracker HAAR classifiers.
     * @return The classifiers array
     */
    public getClassifiers(): Float64Array | Float64Array[] | undefined {
        return this.classifiers;
    }

    /**
     * Gets the edges density value.
     * @return The edges density value
     */
    public getEdgesDensity(): number {
        return this.edgesDensity;
    }

    /**
     * Gets the initial scale to start the feature block scaling.
     * @return The initial scale value
     */
    public getInitialScale(): number {
        return this.initialScale;
    }

    /**
     * Gets the scale factor to scale the feature block.
     * @return The scale factor value
     */
    public getScaleFactor(): number {
        return this.scaleFactor;
    }

    /**
     * Gets the block step size.
     * @return The step size value
     */
    public getStepSize(): number {
        return this.stepSize;
    }

    /**
     * Tracks the Video frames. This method is called for each video frame in
     * order to emit 'track' event.
     * @param pixels The pixels data to track
     * @param width The pixels canvas width
     * @param height The pixels canvas height
     */
    public track(pixels: Uint8ClampedArray, width: number, height: number): void {
        const classifiers = this.getClassifiers();

        if (!classifiers) {
            throw new Error('Object classifier not specified, try `new tracking.ObjectTracker("face")`.');
        }

        let results: Rect[] = [];

        if (Array.isArray(classifiers)) {
            for (const classifier of classifiers) {
                results = results.concat(
                    ViolaJones.detect(
                        pixels,
                        width,
                        height,
                        this.getInitialScale(),
                        this.getScaleFactor(),
                        this.getStepSize(),
                        this.getEdgesDensity(),
                        classifier
                    )
                );
            }
        }

        this.emit("track", {
            data: results,
        });
    }

    /**
     * Sets the tracker HAAR classifiers.
     * @param classifiers The classifiers to set
     */
    public setClassifiers(classifiers: Float64Array[]): void {
        this.classifiers = classifiers;
    }

    /**
     * Sets the edges density.
     * @param edgesDensity The edges density value
     */
    public setEdgesDensity(edgesDensity: number): void {
        this.edgesDensity = edgesDensity;
    }

    /**
     * Sets the initial scale to start the block scaling.
     * @param initialScale The initial scale value
     */
    public setInitialScale(initialScale: number): void {
        this.initialScale = initialScale;
    }

    /**
     * Sets the scale factor to scale the feature block.
     * @param scaleFactor The scale factor value
     */
    public setScaleFactor(scaleFactor: number): void {
        this.scaleFactor = scaleFactor;
    }

    /**
     * Sets the block step size.
     * @param stepSize The step size value
     */
    public setStepSize(stepSize: number): void {
        this.stepSize = stepSize;
    }
}
