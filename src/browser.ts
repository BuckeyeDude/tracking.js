import { TrackerTask } from "./trackers/TrackerTask";
import { Canvas } from "./utils/Canvas";
import { Tracker } from "./trackers/Tracker";

interface TrackingOptions {
    camera?: boolean;
    audio?: boolean;
}

/**
 * Captures the user camera when tracking a video element and set its source
 * to the camera stream.
 * @param {HTMLVideoElement} element Canvas element to track.
 * @param {object} opt_options Optional configuration to the tracker.
 */
function initUserMedia_(element: HTMLVideoElement, opt_options?: TrackingOptions): void {
    window.navigator.mediaDevices
        .getUserMedia({
            video: true,
            audio: !!(opt_options && opt_options.audio),
        })
        .then((stream: MediaStream): void => {
            element.srcObject = stream;
        })
        .catch((): void => {
            throw Error("Cannot capture user camera.");
        });
}

/**
 * Tests whether the object is a dom node.
 * @param {object} o Object to be tested.
 * @return {boolean} True if the object is a dom node.
 */
function isNode(o: any): o is HTMLElement {
    return o.nodeType || isWindow(o);
}

/**
 * Tests whether the object is the `window` object.
 * @param {object} o Object to be tested.
 * @return {boolean} True if the object is the `window` object.
 */
function isWindow(o: any): o is Window {
    return !!(o && o.alert && o.document);
}

/**
 * Selects a dom node from a CSS3 selector using `document.querySelector`.
 * @param {string} selector
 * @param {object} opt_element The root element for the query. When not specified `document` is used as root element.
 * @return {HTMLElement} The first dom element that matches to the selector.
 *     If not found, returns `null`.
 */
export function one(selector: string | HTMLElement, opt_element?: Document | Element): HTMLElement | null {
    if (isNode(selector)) {
        return selector;
    }
    return (opt_element || document).querySelector(selector);
}

/**
 * Tracks a canvas, image or video element based on the specified `tracker`
 * instance. This method extract the pixel information of the input element
 * to pass to the `tracker` instance. When tracking a video, the
 * `tracker.track(pixels, width, height)` will be in a
 * `requestAnimationFrame` loop in order to track all video frames.
 *
 * Example:
 * var tracker = new tracking.ColorTracker();
 *
 * tracking.track('#video', tracker);
 * or
 * tracking.track('#video', tracker, { camera: true });
 *
 * tracker.on('track', function(event) {
 *   // console.log(event.data[0].x, event.data[0].y)
 * });
 *
 * @param {HTMLElement} element The element to track, canvas, image or
 *     video.
 * @param {Tracker} tracker The tracker instance used to track the
 *     element.
 * @param {object} opt_options Optional configuration to the tracker.
 */
export function track(element: string | HTMLElement, tracker: Tracker, opt_options?: TrackingOptions): any {
    const resolvedElement = one(element);
    if (!resolvedElement) {
        throw new Error("Element not found, try a different element or selector.");
    }
    if (!tracker) {
        throw new Error("Tracker not specified, try `tracking.track(element, new tracking.FaceTracker())`.");
    }

    switch (resolvedElement.nodeName.toLowerCase()) {
        case "canvas":
            return trackCanvas_(resolvedElement as HTMLCanvasElement, tracker);
        case "img":
            return trackImg_(resolvedElement as HTMLImageElement, tracker);
        case "video":
            if (opt_options) {
                if (opt_options.camera) {
                    initUserMedia_(resolvedElement as HTMLVideoElement, opt_options);
                }
            }
            return trackVideo_(resolvedElement as HTMLVideoElement, tracker);
        default:
            throw new Error("Element not supported, try in a canvas, img, or video.");
    }
}

/**
 * Tracks a canvas element based on the specified `tracker` instance and
 * returns a `TrackerTask` for this track.
 * @param {HTMLCanvasElement} element Canvas element to track.
 * @param {Tracker} tracker The tracker instance used to track the
 *     element.
 * @return {TrackerTask}
 */
function trackCanvas_(element: HTMLCanvasElement, tracker: Tracker): any {
    const task = new TrackerTask(tracker);
    task.on("run", () => {
        trackCanvasInternal_(element, tracker);
    });
    return task.run();
}

/**
 * Tracks a canvas element based on the specified `tracker` instance. This
 * method extract the pixel information of the input element to pass to the
 * `tracker` instance.
 * @param {HTMLCanvasElement} element Canvas element to track.
 * @param {Tracker} tracker The tracker instance used to track the
 *     element.
 */
function trackCanvasInternal_(element: HTMLCanvasElement, tracker: Tracker): void {
    const width = element.width;
    const height = element.height;
    const context = element.getContext("2d")!;
    const imageData = context.getImageData(0, 0, width, height);
    tracker.track(imageData.data, width, height);
}

/**
 * Tracks a image element based on the specified `tracker` instance. This
 * method extract the pixel information of the input element to pass to the
 * `tracker` instance.
 * @param {HTMLImageElement} element Canvas element to track.
 * @param {Tracker} tracker The tracker instance used to track the
 *     element.
 */
function trackImg_(element: HTMLImageElement, tracker: Tracker): any {
    const width = element.naturalWidth;
    const height = element.naturalHeight;
    const canvas = document.createElement("canvas");

    canvas.width = width;
    canvas.height = height;

    const task = new TrackerTask(tracker);
    task.on("run", (): void => {
        Canvas.loadImage(canvas, element.src, 0, 0, width, height, (): void => {
            trackCanvasInternal_(canvas, tracker);
        });
    });
    return task.run();
}

/**
 * Tracks a video element based on the specified `tracker` instance. This
 * method extract the pixel information of the input element to pass to the
 * `tracker` instance. The `tracker.track(pixels, width, height)` will be in
 * a `requestAnimationFrame` loop in order to track all video frames.
 * @param {HTMLVideoElement} element Canvas element to track.
 * @param {Tracker} tracker The tracker instance used to track the
 *     element.
 * @private
 */
function trackVideo_(element: HTMLVideoElement, tracker: Tracker): any {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d")!;
    let width: number;
    let height: number;

    // FIXME here the video display size of the analysed size
    const resizeCanvas_ = function (): void {
        width = element.offsetWidth;
        height = element.offsetHeight;
        canvas.width = width;
        canvas.height = height;
    };
    resizeCanvas_();
    element.addEventListener("resize", resizeCanvas_);

    // FIXME: do a process function - it is up to the caller to handle the frequency of detection
    // it seems all handled in the tracking.TrackerTask..
    // so in short, remove the tracking.TrackerTask from here
    // if the user want to use it, it can create it himself
    let requestId: number;
    const requestAnimationFrame_ = function (): void {
        requestId = window.requestAnimationFrame(function (): void {
            if (element.readyState === element.HAVE_ENOUGH_DATA) {
                try {
                    // Firefox v~30.0 gets confused with the video readyState firing an
                    // erroneous HAVE_ENOUGH_DATA just before HAVE_CURRENT_DATA state,
                    // hence keep trying to read it until resolved.
                    context.drawImage(element, 0, 0, width, height);
                } catch (err) {}
                trackCanvasInternal_(canvas, tracker);
            }
            requestAnimationFrame_();
        });
    };

    const task = new TrackerTask(tracker);
    task.on("stop", function (): void {
        window.cancelAnimationFrame(requestId);
    });
    task.on("run", function (): void {
        requestAnimationFrame_();
    });
    return task.run();
}
