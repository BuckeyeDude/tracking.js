import { EventEmitter } from "eventemitter3";

import { Rect } from "../detection/Rect";

interface Events {
    track: {
        data: Rect[];
    };
}

export abstract class Tracker extends EventEmitter<Events> {
    /**
     * Tracks the pixels on the array. This method is called for each video
     * frame in order to emit `track` event.
     * @param {Uint8ClampedArray} pixels The pixels data to track.
     * @param {number} width The pixels canvas width.
     * @param {number} height The pixels canvas height.
     */
    abstract track(pixels: Uint8ClampedArray, width: number, height: number): void;
}
