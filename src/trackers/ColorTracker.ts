import { Tracker } from "./Tracker";
import { Rect } from "../detection/Rect";

/**
 * Interface for color detection function
 */
type ColorFunction = (r: number, g: number, b: number, a?: number, w?: number, i?: number, j?: number) => boolean;


/**
 * ColorTracker utility to track colored blobs in a frame using color
 * difference evaluation.
 */
export class ColorTracker extends Tracker {
    private static knownColors_: Record<string, ColorFunction> = {};
    private static neighbours_: Record<number, Int32Array> = {};

    /**
     * Holds the colors to be tracked by the ColorTracker instance.
     */
    public colors: string[] = ['magenta'];

    /**
     * Holds the minimum dimension to classify a rectangle.
     */
    public minDimension: number = 20;

    /**
     * Holds the maximum dimension to classify a rectangle.
     */
    public maxDimension: number = Infinity;

    /**
     * Holds the minimum group size to be classified as a rectangle.
     */
    public minGroupSize: number = 30;

    /**
     * Creates a new ColorTracker instance
     * @param opt_colors Optional colors to track
     */
    constructor(opt_colors?: string | string[]) {
        super();

        if (typeof opt_colors === 'string') {
            opt_colors = [opt_colors];
        }

        if (opt_colors) {
            opt_colors.forEach((color) => {
                if (!ColorTracker.getColor(color)) {
                    throw new Error('Color not valid, try `new ColorTracker("magenta")`.');
                }
            });
            this.setColors(opt_colors);
        }
    }

    /**
     * Registers a color as known color.
     * @param name The color name
     * @param fn The color function to test if the passed (r,g,b) is the desired color
     */
    public static registerColor(name: string, fn: ColorFunction): void {
        ColorTracker.knownColors_[name] = fn;
    }

    /**
     * Gets the known color function that is able to test whether an (r,g,b) is
     * the desired color.
     * @param name The color name
     * @returns The known color test function
     */
    public static getColor(name: string): ColorFunction | undefined {
        return ColorTracker.knownColors_[name];
    }

    /**
     * Calculates the central coordinate from the cloud points. The cloud points
     * are all points that matches the desired color.
     * @param cloud Major row order array containing all the points from the desired color
     * @param total Total numbers of pixels of the desired color
     * @returns Object containing the x, y coordinates and dimensions of the blob
     */
    private calculateDimensions_(cloud: Int32Array, total: number): Rect | null {
        let maxx = -1;
        let maxy = -1;
        let minx = Infinity;
        let miny = Infinity;

        for (let c = 0; c < total; c += 2) {
            const x = cloud[c];
            const y = cloud[c + 1];

            if (x < minx) {
                minx = x;
            }
            if (x > maxx) {
                maxx = x;
            }
            if (y < miny) {
                miny = y;
            }
            if (y > maxy) {
                maxy = y;
            }
        }

        return {
            width: maxx - minx,
            height: maxy - miny,
            x: minx,
            y: miny,
            total: 0,
            color: '' // Will be set by caller
        };
    }

    /**
     * Gets the colors being tracked by the ColorTracker instance.
     */
    public getColors(): string[] {
        return this.colors;
    }

    /**
     * Gets the minimum dimension to classify a rectangle.
     */
    public getMinDimension(): number {
        return this.minDimension;
    }

    /**
     * Gets the maximum dimension to classify a rectangle.
     */
    public getMaxDimension(): number {
        return this.maxDimension;
    }

    /**
     * Gets the minimum group size to be classified as a rectangle.
     */
    public getMinGroupSize(): number {
        return this.minGroupSize;
    }

    /**
     * Gets the eight offset values of the neighbours surrounding a pixel.
     * @param width The image width
     * @returns Array with the eight offset values of the neighbours surrounding a pixel
     */
    private getNeighboursForWidth_(width: number): Int32Array {
        if (ColorTracker.neighbours_[width]) {
            return ColorTracker.neighbours_[width];
        }

        const neighbours = new Int32Array(8);

        neighbours[0] = -width * 4;
        neighbours[1] = -width * 4 + 4;
        neighbours[2] = 4;
        neighbours[3] = width * 4 + 4;
        neighbours[4] = width * 4;
        neighbours[5] = width * 4 - 4;
        neighbours[6] = -4;
        neighbours[7] = -width * 4 - 4;

        ColorTracker.neighbours_[width] = neighbours;

        return neighbours;
    }

    /**
     * Unites groups whose bounding box intersect with each other.
     * @param rects Array of rectangles to merge
     */
    private mergeRectangles_(rects: Rect[]): Rect[] {
        let intersects: boolean;
        const results: Rect[] = [];
        const minDimension = this.getMinDimension();
        const maxDimension = this.getMaxDimension();

        for (let r = 0; r < rects.length; r++) {
            const r1 = rects[r];
            intersects = true;
            for (let s = r + 1; s < rects.length; s++) {
                const r2 = rects[s];
                if (this.intersectRect_(r1.x, r1.y, r1.x + r1.width, r1.y + r1.height, r2.x, r2.y, r2.x + r2.width, r2.y + r2.height)) {
                    intersects = false;
                    const x1 = Math.min(r1.x, r2.x);
                    const y1 = Math.min(r1.y, r2.y);
                    const x2 = Math.max(r1.x + r1.width, r2.x + r2.width);
                    const y2 = Math.max(r1.y + r1.height, r2.y + r2.height);
                    r2.height = y2 - y1;
                    r2.width = x2 - x1;
                    r2.x = x1;
                    r2.y = y1;
                    break;
                }
            }

            if (intersects) {
                if (r1.width >= minDimension && r1.height >= minDimension) {
                    if (r1.width <= maxDimension && r1.height <= maxDimension) {
                        results.push(r1);
                    }
                }
            }
        }

        return results;
    }

    /**
     * Helper method to check if two rectangles intersect
     */
    private intersectRect_(x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, x4: number, y4: number): boolean {
        return !(x2 < x3 || x4 < x1 || y2 < y3 || y4 < y1);
    }

    /**
     * Sets the colors to be tracked by the ColorTracker instance.
     */
    public setColors(colors: string[]): void {
        this.colors = colors;
    }

    /**
     * Sets the minimum dimension to classify a rectangle.
     */
    public setMinDimension(minDimension: number): void {
        this.minDimension = minDimension;
    }

    /**
     * Sets the maximum dimension to classify a rectangle.
     */
    public setMaxDimension(maxDimension: number): void {
        this.maxDimension = maxDimension;
    }

    /**
     * Sets the minimum group size to be classified as a rectangle.
     */
    public setMinGroupSize(minGroupSize: number): void {
        this.minGroupSize = minGroupSize;
    }

    /**
     * Tracks the Video frames. This method is called for each video frame in
     * order to emit 'track' event.
     * @param pixels The pixels data to track
     * @param width The pixels canvas width
     * @param height The pixels canvas height
     */
    public track(pixels: Uint8ClampedArray, width: number, height: number): void {
        const colors = this.getColors();

        if (!colors) {
            throw new Error('Colors not specified, try `new ColorTracker("magenta")`.');
        }

        let results: Rect[] = [];

        colors.forEach((color) => {
            results = results.concat(this.trackColor_(pixels, width, height, color));
        });

        this.emit('track', {
            data: results
        });
    }

    /**
     * Find the given color in the given matrix of pixels using Flood fill
     * algorithm to determines the area connected to a given node in a
     * multi-dimensional array.
     * @param pixels The pixels data to track
     * @param width The pixels canvas width
     * @param height The pixels canvas height
     * @param color The color to be found
     */
    private trackColor_(pixels: Uint8ClampedArray, width: number, height: number, color: string): Rect[] {
        const colorFn = ColorTracker.knownColors_[color];
        const currGroup = new Int32Array(pixels.length >> 2);
        let currGroupSize: number;
        let currI: number;
        let currJ: number;
        let currW: number;
        const marked = new Int8Array(pixels.length);
        const minGroupSize = this.getMinGroupSize();
        const neighboursW = this.getNeighboursForWidth_(width);
        const queue = new Int32Array(pixels.length);
        let queuePosition: number;
        const results: Rect[] = [];
        let w = -4;

        // Caching neighbour i/j offset values
        const neighboursI = new Int32Array([-1, -1, 0, 1, 1, 1, 0, -1]);
        const neighboursJ = new Int32Array([0, 1, 1, 1, 0, -1, -1, -1]);

        if (!colorFn) {
            return results;
        }

        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                w += 4;

                if (marked[w]) {
                    continue;
                }

                currGroupSize = 0;

                queuePosition = -1;
                queue[++queuePosition] = w;
                queue[++queuePosition] = i;
                queue[++queuePosition] = j;

                marked[w] = 1;

                while (queuePosition >= 0) {
                    currJ = queue[queuePosition--];
                    currI = queue[queuePosition--];
                    currW = queue[queuePosition--];

                    if (colorFn(pixels[currW], pixels[currW + 1], pixels[currW + 2], pixels[currW + 3], currW, currI, currJ)) {
                        currGroup[currGroupSize++] = currJ;
                        currGroup[currGroupSize++] = currI;

                        for (let k = 0; k < neighboursW.length; k++) {
                            const otherW = currW + neighboursW[k];
                            const otherI = currI + neighboursI[k];
                            const otherJ = currJ + neighboursJ[k];
                            if (!marked[otherW] && otherI >= 0 && otherI < height && otherJ >= 0 && otherJ < width) {
                                queue[++queuePosition] = otherW;
                                queue[++queuePosition] = otherI;
                                queue[++queuePosition] = otherJ;

                                marked[otherW] = 1;
                            }
                        }
                    }
                }

                if (currGroupSize >= minGroupSize) {
                    const data = this.calculateDimensions_(currGroup, currGroupSize);
                    if (data) {
                        data.color = color;
                        results.push(data);
                    }
                }
            }
        }

        return this.mergeRectangles_(results);
    }
}

// Register default colors
ColorTracker.registerColor('cyan', (r: number, g: number, b: number): boolean => {
    const thresholdGreen = 50;
    const thresholdBlue = 70;
    const dx = r - 0;
    const dy = g - 255;
    const dz = b - 255;

    if ((g - r) >= thresholdGreen && (b - r) >= thresholdBlue) {
        return true;
    }
    return dx * dx + dy * dy + dz * dz < 6400;
});

ColorTracker.registerColor('magenta', (r: number, g: number, b: number): boolean => {
    const threshold = 50;
    const dx = r - 255;
    const dy = g - 0;
    const dz = b - 255;

    if ((r - g) >= threshold && (b - g) >= threshold) {
        return true;
    }
    return dx * dx + dy * dy + dz * dz < 19600;
});

ColorTracker.registerColor('yellow', (r: number, g: number, b: number): boolean => {
    const threshold = 50;
    const dx = r - 255;
    const dy = g - 255;
    const dz = b - 0;

    if ((r - b) >= threshold && (g - b) >= threshold) {
        return true;
    }
    return dx * dx + dy * dy + dz * dz < 10000;
});
