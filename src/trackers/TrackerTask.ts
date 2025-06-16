import { EventEmitter } from "eventemitter3";
import { Tracker } from "./Tracker";

export class TrackerTask extends EventEmitter {
    /**
     * Holds the tracker instance managed by this task.
     */
    private tracker_: Tracker;

    /**
     * Holds if the tracker task is in running.
     */
    private running_: boolean = false;

    /**
     * Function to re-emit track events
     * @private
     */
    private reemitTrackEvent_?: (event: any) => void;

    /**
     * Creates a new TrackerTask instance.
     * @param tracker - The tracker instance to manage
     */
    constructor(tracker: Tracker) {
        super();

        if (!tracker) {
            throw new Error("Tracker instance not specified.");
        }

        this.tracker_ = tracker;
    }

    /**
     * Gets the tracker instance managed by this task.
     * @return The tracker instance
     */
    getTracker(): Tracker {
        return this.tracker_;
    }

    /**
     * Returns true if the tracker task is in running, false otherwise.
     * @return Whether the task is running
     * @private
     */
    inRunning(): boolean {
        return this.running_;
    }

    /**
     * Sets if the tracker task is in running.
     * @param running - Whether the task should be running
     * @private
     */
    setRunning(running: boolean): void {
        this.running_ = running;
    }

    /**
     * Sets the tracker instance managed by this task.
     * @param tracker - The tracker instance to set
     */
    setTracker(tracker: Tracker): void {
        this.tracker_ = tracker;
    }

    /**
     * Emits a `run` event on the tracker task for the implementers to run any
     * child action, e.g. `requestAnimationFrame`.
     * @return Returns itself, so calls can be chained.
     */
    run(): this {
        if (this.inRunning()) {
            return this;
        }

        this.setRunning(true);
        this.reemitTrackEvent_ = (event: any) => {
            this.emit("track", event);
        };
        this.tracker_.on("track", this.reemitTrackEvent_);
        this.emit("run");
        return this;
    }

    /**
     * Emits a `stop` event on the tracker task for the implementers to stop any
     * child action being done, e.g. `requestAnimationFrame`.
     * @return Returns itself, so calls can be chained.
     */
    stop(): this {
        if (!this.inRunning()) {
            return this;
        }

        this.setRunning(false);
        this.emit("stop");
        if (this.reemitTrackEvent_) {
            this.tracker_.removeListener("track", this.reemitTrackEvent_);
        }
        return this;
    }
}
