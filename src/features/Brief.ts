/**
 * Brief intends for "Binary Robust Independent Elementary Features". This
 * class generates a binary string for each keypoint found by an extractor
 * method.
 */
export class Brief {
    /**
     * The set of binary tests is defined by the nd (x,y)-location pairs
     * uniquely chosen during the initialization. Values could vary between N =
     * 128,256,512. N=128 yield good compromises between speed, storage
     * efficiency, and recognition rate.
     */
    public static readonly N: number = 512;

    /**
     * Caches coordinates values of (x,y)-location pairs uniquely chosen during
     * the initialization.
     */
    private static randomImageOffsets_: { [key: number]: Int32Array } = {};

    /**
     * Caches delta values of (x,y)-location pairs uniquely chosen during
     * the initialization.
     */
    private static randomWindowOffsets_: Int32Array | null = null;

    /**
     * Generates a binary string for each found keypoints extracted using an
     * extractor method.
     * @param pixels The grayscale pixels in a linear [p1,p2,...] array.
     * @param width The image width.
     * @param keypoints Array of keypoint coordinates
     * @return Returns an array where for each four sequence int
     *     values represent the descriptor binary string (128 bits) necessary
     *     to describe the corner, e.g. [0,0,0,0, 0,0,0,0, ...].
     */
    public static getDescriptors(pixels: number[], width: number, keypoints: number[]): Int32Array {
        // Optimizing divide by 32 operation using binary shift
        // (this.N >> 5) === this.N/32.
        const descriptors = new Int32Array((keypoints.length >> 1) * (this.N >> 5));
        let descriptorWord = 0;
        const offsets = this.getRandomOffsets_(width);
        let position = 0;

        for (let i = 0; i < keypoints.length; i += 2) {
            const w = width * keypoints[i + 1] + keypoints[i];

            let offsetsPosition = 0;
            for (let j = 0, n = this.N; j < n; j++) {
                if (pixels[offsets[offsetsPosition++] + w] < pixels[offsets[offsetsPosition++] + w]) {
                    // The bit in the position `j % 32` of descriptorWord should be set to 1. We do
                    // this by making an OR operation with a binary number that only has the bit
                    // in that position set to 1. That binary number is obtained by shifting 1 left by
                    // `j % 32` (which is the same as `j & 31` left) positions.
                    descriptorWord |= 1 << (j & 31);
                }

                // If the next j is a multiple of 32, we will need to use a new descriptor word to hold
                // the next results.
                if (!((j + 1) & 31)) {
                    descriptors[position++] = descriptorWord;
                    descriptorWord = 0;
                }
            }
        }

        return descriptors;
    }

    /**
     * Matches sets of features {mi} and {m′j} extracted from two images taken
     * from similar, and often successive, viewpoints. A classical procedure
     * runs as follows. For each point {mi} in the first image, search in a
     * region of the second image around location {mi} for point {m′j}. The
     * search is based on the similarity of the local image windows, also known
     * as kernel windows, centered on the points, which strongly characterizes
     * the points when the images are sufficiently close. Once each keypoint is
     * described with its binary string, they need to be compared with the
     * closest matching point. Distance metric is critical to the performance of
     * in- trusion detection systems. Thus using binary strings reduces the size
     * of the descriptor and provides an interesting data structure that is fast
     * to operate whose similarity can be measured by the Hamming distance.
     * @param keypoints1 First set of keypoints
     * @param descriptors1 First set of descriptors
     * @param keypoints2 Second set of keypoints
     * @param descriptors2 Second set of descriptors
     * @return Returns an array where the index is the corner1
     *     index coordinate, and the value is the corresponding match index of
     *     corner2, e.g. keypoints1=[x0,y0,x1,y1,...] and
     *     keypoints2=[x'0,y'0,x'1,y'1,...], if x0 matches x'1 and x1 matches x'0,
     *     the return array would be [3,0].
     */
    public static match(
        keypoints1: number[],
        descriptors1: Int32Array,
        keypoints2: number[],
        descriptors2: Int32Array
    ): MatchResult[] {
        const len1 = keypoints1.length >> 1;
        const len2 = keypoints2.length >> 1;
        const matches: MatchResult[] = new Array(len1);

        for (let i = 0; i < len1; i++) {
            let min = Infinity;
            let minj = 0;
            for (let j = 0; j < len2; j++) {
                let dist = 0;
                // Optimizing divide by 32 operation using binary shift
                // (this.N >> 5) === this.N/32.
                for (let k = 0, n = this.N >> 5; k < n; k++) {
                    dist += this.hammingWeight(descriptors1[i * n + k] ^ descriptors2[j * n + k]);
                }
                if (dist < min) {
                    min = dist;
                    minj = j;
                }
            }
            matches[i] = {
                index1: i,
                index2: minj,
                keypoint1: [keypoints1[2 * i], keypoints1[2 * i + 1]],
                keypoint2: [keypoints2[2 * minj], keypoints2[2 * minj + 1]],
                confidence: 1 - min / this.N,
            };
        }

        return matches;
    }

    /**
     * Removes matches outliers by testing matches on both directions.
     * @param keypoints1 First set of keypoints
     * @param descriptors1 First set of descriptors
     * @param keypoints2 Second set of keypoints
     * @param descriptors2 Second set of descriptors
     * @return Returns an array where the index is the corner1
     *     index coordinate, and the value is the corresponding match index of
     *     corner2, e.g. keypoints1=[x0,y0,x1,y1,...] and
     *     keypoints2=[x'0,y'0,x'1,y'1,...], if x0 matches x'1 and x1 matches x'0,
     *     the return array would be [3,0].
     */
    public static reciprocalMatch(
        keypoints1: number[],
        descriptors1: Int32Array,
        keypoints2: number[],
        descriptors2: Int32Array
    ): MatchResult[] {
        const matches: MatchResult[] = [];
        if (keypoints1.length === 0 || keypoints2.length === 0) {
            return matches;
        }

        const matches1 = this.match(keypoints1, descriptors1, keypoints2, descriptors2);
        const matches2 = this.match(keypoints2, descriptors2, keypoints1, descriptors1);
        for (let i = 0; i < matches1.length; i++) {
            if (matches2[matches1[i].index2].index2 === i) {
                matches.push(matches1[i]);
            }
        }
        return matches;
    }

    /**
     * Gets the coordinates values of (x,y)-location pairs uniquely chosen
     * during the initialization.
     * @param width Image width
     * @return Array with the random offset values.
     */
    private static getRandomOffsets_(width: number): Int32Array {
        if (!this.randomWindowOffsets_) {
            let windowPosition = 0;
            const windowOffsets = new Int32Array(4 * this.N);
            for (let i = 0; i < this.N; i++) {
                windowOffsets[windowPosition++] = Math.round(this.uniformRandom(-15, 16));
                windowOffsets[windowPosition++] = Math.round(this.uniformRandom(-15, 16));
                windowOffsets[windowPosition++] = Math.round(this.uniformRandom(-15, 16));
                windowOffsets[windowPosition++] = Math.round(this.uniformRandom(-15, 16));
            }
            this.randomWindowOffsets_ = windowOffsets;
        }

        if (!this.randomImageOffsets_[width]) {
            let imagePosition = 0;
            const imageOffsets = new Int32Array(2 * this.N);
            for (let j = 0; j < this.N; j++) {
                imageOffsets[imagePosition++] =
                    this.randomWindowOffsets_[4 * j] * width + this.randomWindowOffsets_[4 * j + 1];
                imageOffsets[imagePosition++] =
                    this.randomWindowOffsets_[4 * j + 2] * width + this.randomWindowOffsets_[4 * j + 3];
            }
            this.randomImageOffsets_[width] = imageOffsets;
        }

        return this.randomImageOffsets_[width];
    }

    /**
     * Placeholder for hamming weight calculation - you'll need to implement this
     * or import it from your math utilities
     */
    private static hammingWeight(value: number): number {
        // You'll need to implement this method or import it from tracking.Math
        // This is a placeholder implementation
        let count = 0;
        while (value) {
            count += value & 1;
            value >>>= 1;
        }
        return count;
    }

    /**
     * Placeholder for uniform random number generation - you'll need to implement this
     * or import it from your math utilities
     */
    private static uniformRandom(min: number, max: number): number {
        // You'll need to implement this method or import it from tracking.Math
        // This is a placeholder implementation
        return Math.random() * (max - min) + min;
    }
}

/**
 * Interface for match results
 */
export interface MatchResult {
    index1: number;
    index2: number;
    keypoint1: [number, number];
    keypoint2: [number, number];
    confidence: number;
}
