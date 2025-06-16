/**
 * Image utility class for various image processing operations.
 */
export class Image {
    /**
     * Computes gaussian blur. Adapted from
     * https://github.com/kig/canvasfilters.
     * @param pixels The pixels in a linear [r,g,b,a,...] array.
     * @param width The image width.
     * @param height The image height.
     * @param diameter Gaussian blur diameter, must be greater than 1.
     * @return The edge pixels in a linear [r,g,b,a,...] array.
     */
    static blur(pixels: Uint8Array, width: number, height: number, diameter: number): Float32Array {
        diameter = Math.abs(diameter);
        if (diameter <= 1) {
            throw new Error("Diameter should be greater than 1.");
        }
        const radius = diameter / 2;
        const len = Math.ceil(diameter) + (1 - (Math.ceil(diameter) % 2));
        const weights = new Float32Array(len);
        const rho = (radius + 0.5) / 3;
        const rhoSq = rho * rho;
        const gaussianFactor = 1 / Math.sqrt(2 * Math.PI * rhoSq);
        const rhoFactor = -1 / (2 * rho * rho);
        let wsum = 0;
        const middle = Math.floor(len / 2);

        for (let i = 0; i < len; i++) {
            const x = i - middle;
            const gx = gaussianFactor * Math.exp(x * x * rhoFactor);
            weights[i] = gx;
            wsum += gx;
        }

        for (let j = 0; j < weights.length; j++) {
            weights[j] /= wsum;
        }

        return this.separableConvolve(pixels, width, height, weights, weights, false);
    }

    /**
     * Computes the integral image for summed, squared, rotated and sobel pixels.
     * @param pixels The pixels in a linear [r,g,b,a,...] array to loop through.
     * @param width The image width.
     * @param height The image height.
     * @param optIntegralImage Empty array of size `width * height` to be filled with the integral image values.
     * @param optIntegralImageSquare Empty array of size `width * height` to be filled with the integral image squared values.
     * @param optTiltedIntegralImage Empty array of size `width * height` to be filled with the rotated integral image values.
     * @param optIntegralImageSobel Empty array of size `width * height` to be filled with the integral image of sobel values.
     */
    static computeIntegralImage(
        pixels: Uint8ClampedArray,
        width: number,
        height: number,
        optIntegralImage?: Int32Array,
        optIntegralImageSquare?: Int32Array,
        optTiltedIntegralImage?: Int32Array,
        optIntegralImageSobel?: Int32Array
    ): void {
        if (arguments.length < 4) {
            throw new Error("You should specify at least one output array in the order: sum, square, tilted, sobel.");
        }

        let pixelsSobel: Float32Array | undefined;
        if (optIntegralImageSobel) {
            pixelsSobel = this.sobel(pixels, width, height);
        }

        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                const w = i * width * 4 + j * 4;
                const pixel = ~~(pixels[w] * 0.299 + pixels[w + 1] * 0.587 + pixels[w + 2] * 0.114);

                if (optIntegralImage) {
                    this.computePixelValueSAT_(optIntegralImage, width, i, j, pixel);
                }

                if (optIntegralImageSquare) {
                    this.computePixelValueSAT_(optIntegralImageSquare, width, i, j, pixel * pixel);
                }

                if (optTiltedIntegralImage) {
                    const w1 = w - width * 4;
                    const pixelAbove = ~~(pixels[w1] * 0.299 + pixels[w1 + 1] * 0.587 + pixels[w1 + 2] * 0.114);
                    this.computePixelValueRSAT_(optTiltedIntegralImage, width, i, j, pixel, pixelAbove || 0);
                }

                if (optIntegralImageSobel && pixelsSobel) {
                    this.computePixelValueSAT_(optIntegralImageSobel, width, i, j, pixelsSobel[w]);
                }
            }
        }
    }

    /**
     * Helper method to compute the rotated summed area table (RSAT) by the formula:
     * RSAT(x, y) = RSAT(x-1, y-1) + RSAT(x+1, y-1) - RSAT(x, y-2) + I(x, y) + I(x, y-1)
     */
    private static computePixelValueRSAT_(
        RSAT: Int32Array,
        width: number,
        i: number,
        j: number,
        pixel: number,
        pixelAbove: number
    ): void {
        const w = i * width + j;
        RSAT[w] =
            (RSAT[w - width - 1] || 0) +
            (RSAT[w - width + 1] || 0) -
            (RSAT[w - width - width] || 0) +
            pixel +
            pixelAbove;
    }

    /**
     * Helper method to compute the summed area table (SAT) by the formula:
     * SAT(x, y) = SAT(x, y-1) + SAT(x-1, y) + I(x, y) - SAT(x-1, y-1)
     */
    private static computePixelValueSAT_(SAT: Int32Array, width: number, i: number, j: number, pixel: number): void {
        const w = i * width + j;
        SAT[w] = (SAT[w - width] || 0) + (SAT[w - 1] || 0) + pixel - (SAT[w - width - 1] || 0);
    }

    /**
     * Converts a color from a color-space based on an RGB color model to a
     * grayscale representation of its luminance.
     * @param pixels The pixels in a linear [r,g,b,a,...] array.
     * @param width The image width.
     * @param height The image height.
     * @param fillRGBA If the result should fill all RGBA values with the gray scale values.
     * @return The grayscale pixels in a linear array.
     */
    static grayscale(pixels: Uint8ClampedArray, width: number, height: number, fillRGBA: boolean = false): Uint8Array {
        const len = pixels.length >> 2;
        const gray = fillRGBA ? new Uint32Array(len) : new Uint8Array(len);
        const data32 = new Uint32Array(pixels.buffer || new Uint8Array(pixels).buffer);
        let i = 0;
        let c = 0;
        let luma = 0;

        // unrolled loops to not have to check fillRGBA each iteration
        if (fillRGBA) {
            while (i < len) {
                // Entire pixel in little-endian order (ABGR)
                c = data32[i];
                luma = (((c >>> 16) & 0xff) * 13933 + ((c >>> 8) & 0xff) * 46871 + (c & 0xff) * 4732) >>> 16;
                gray[i++] = (luma * 0x10101) | (c & 0xff000000);
            }
        } else {
            while (i < len) {
                c = data32[i];
                luma = (((c >>> 16) & 0xff) * 13933 + ((c >>> 8) & 0xff) * 46871 + (c & 0xff) * 4732) >>> 16;
                gray[i++] = luma;
            }
        }

        return new Uint8Array(gray.buffer);
    }

    /**
     * Fast horizontal separable convolution.
     * @param pixels The pixels in a linear [r,g,b,a,...] array.
     * @param width The image width.
     * @param height The image height.
     * @param weightsVector The weighting vector, e.g [-1,0,1].
     * @param opaque Opacity flag.
     * @return The convoluted pixels in a linear [r,g,b,a,...] array.
     */
    static horizontalConvolve(
        pixels: Float32Array,
        width: number,
        height: number,
        weightsVector: Float32Array,
        opaque: boolean
    ): Float32Array {
        const side = weightsVector.length;
        const halfSide = Math.floor(side / 2);
        const output = new Float32Array(width * height * 4);
        const alphaFac = opaque ? 1 : 0;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const sy = y;
                const sx = x;
                const offset = (y * width + x) * 4;
                let r = 0;
                let g = 0;
                let b = 0;
                let a = 0;

                for (let cx = 0; cx < side; cx++) {
                    const scy = sy;
                    const scx = Math.min(width - 1, Math.max(0, sx + cx - halfSide));
                    const poffset = (scy * width + scx) * 4;
                    const wt = weightsVector[cx];
                    r += pixels[poffset] * wt;
                    g += pixels[poffset + 1] * wt;
                    b += pixels[poffset + 2] * wt;
                    a += pixels[poffset + 3] * wt;
                }

                output[offset] = r;
                output[offset + 1] = g;
                output[offset + 2] = b;
                output[offset + 3] = a + alphaFac * (255 - a);
            }
        }

        return output;
    }

    /**
     * Fast vertical separable convolution.
     * @param pixels The pixels in a linear [r,g,b,a,...] array.
     * @param width The image width.
     * @param height The image height.
     * @param weightsVector The weighting vector, e.g [-1,0,1].
     * @param opaque Opacity flag.
     * @return The convoluted pixels in a linear [r,g,b,a,...] array.
     */
    static verticalConvolve(
        pixels: Uint8Array,
        width: number,
        height: number,
        weightsVector: number[] | Float32Array,
        opaque: boolean
    ): Float32Array {
        const side = weightsVector.length;
        const halfSide = Math.floor(side / 2);
        const output = new Float32Array(width * height * 4);
        const alphaFac = opaque ? 1 : 0;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const sy = y;
                const sx = x;
                const offset = (y * width + x) * 4;
                let r = 0;
                let g = 0;
                let b = 0;
                let a = 0;

                for (let cy = 0; cy < side; cy++) {
                    const scy = Math.min(height - 1, Math.max(0, sy + cy - halfSide));
                    const scx = sx;
                    const poffset = (scy * width + scx) * 4;
                    const wt = weightsVector[cy];
                    r += pixels[poffset] * wt;
                    g += pixels[poffset + 1] * wt;
                    b += pixels[poffset + 2] * wt;
                    a += pixels[poffset + 3] * wt;
                }

                output[offset] = r;
                output[offset + 1] = g;
                output[offset + 2] = b;
                output[offset + 3] = a + alphaFac * (255 - a);
            }
        }

        return output;
    }

    /**
     * Fast separable convolution.
     * @param pixels The pixels in a linear [r,g,b,a,...] array.
     * @param width The image width.
     * @param height The image height.
     * @param horizWeights The horizontal weighting vector, e.g [-1,0,1].
     * @param vertWeights The vertical vector, e.g [-1,0,1].
     * @param opaque Opacity flag.
     * @return The convoluted pixels in a linear [r,g,b,a,...] array.
     */
    static separableConvolve(
        pixels: Uint8Array,
        width: number,
        height: number,
        horizWeights: Float32Array,
        vertWeights: Float32Array,
        opaque: boolean
    ): Float32Array {
        const vertical = this.verticalConvolve(pixels, width, height, vertWeights, opaque);
        return this.horizontalConvolve(vertical, width, height, horizWeights, opaque);
    }

    /**
     * Compute image edges using Sobel operator.
     * @param pixels The pixels in a linear [r,g,b,a,...] array.
     * @param width The image width.
     * @param height The image height.
     * @return The edge pixels in a linear [r,g,b,a,...] array.
     */
    static sobel(pixels: Uint8ClampedArray, width: number, height: number): Float32Array {
        const grayscalePixels = this.grayscale(pixels, width, height, true);
        const output = new Float32Array(width * height * 4);
        const sobelSignVector = new Float32Array([-1, 0, 1]);
        const sobelScaleVector = new Float32Array([1, 2, 1]);
        const vertical = this.separableConvolve(
            grayscalePixels,
            width,
            height,
            sobelSignVector,
            sobelScaleVector,
            false
        );
        const horizontal = this.separableConvolve(
            grayscalePixels,
            width,
            height,
            sobelScaleVector,
            sobelSignVector,
            false
        );

        for (let i = 0; i < output.length; i += 4) {
            const v = vertical[i];
            const h = horizontal[i];
            const p = Math.sqrt(h * h + v * v);
            output[i] = p;
            output[i + 1] = p;
            output[i + 2] = p;
            output[i + 3] = 255;
        }

        return output;
    }

    /**
     * Equalizes the histogram of a grayscale image, normalizing the
     * brightness and increasing the contrast of the image.
     * @param pixels The grayscale pixels in a linear array.
     * @param width The image width.
     * @param height The image height.
     * @return The equalized grayscale pixels in a linear array.
     */
    static equalizeHist(pixels: Uint8Array, width: number, height: number): Uint8ClampedArray {
        const equalized = new Uint8ClampedArray(pixels.length);

        const histogram = new Array<number>(256);
        for (let i = 0; i < 256; i++) histogram[i] = 0;

        for (let i = 0; i < pixels.length; i++) {
            equalized[i] = pixels[i];
            histogram[pixels[i]]++;
        }

        let prev = histogram[0];
        for (let i = 0; i < 256; i++) {
            histogram[i] += prev;
            prev = histogram[i];
        }

        const norm = 255 / pixels.length;
        for (let i = 0; i < pixels.length; i++) {
            equalized[i] = (histogram[pixels[i]] * norm + 0.5) | 0;
        }

        return equalized;
    }
}
