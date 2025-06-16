import { ObjectTracker } from "./ObjectTracker";
import { ViolaJones } from "../detection/ViolaJones";
import { LBF } from "../alignment/LBF";

export class LandmarksTracker extends ObjectTracker {
    track(pixels: Uint8ClampedArray, width: number, height: number) {
        const image = {
            data: pixels,
            width: width,
            height: height,
        };

        const classifier = ViolaJones.classifiers["face"];

        const faces = ViolaJones.detect(
            pixels,
            width,
            height,
            this.getInitialScale(),
            this.getScaleFactor(),
            this.getStepSize(),
            this.getEdgesDensity(),
            classifier
        );

        const landmarks = LBF.align(pixels, width, height, faces);

        this.emit("track", {
            data: {
                faces: faces,
                landmarks: landmarks,
            },
        });
    }
}
