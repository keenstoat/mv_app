export default {
    template: `
    <canvas
      ref="canvas"
      style="display: block;"
    ></canvas>
  `,
    props: {
        src: String,
    },

    mounted() {
        this.canvas = this.$refs.canvas;
        this.canvas_context = this.canvas.getContext("2d");

        this.decoder = new TextDecoder();
        this.latestFrame = null;
        this.rendering = false;
        this.disposed = false;
    },

    beforeUnmount() {
        this.disposed = true;
        this.ws?.close();
    },

    methods: {
        start_stream() {
            this.ws = new WebSocket(this.src);
            this.ws.binaryType = "arraybuffer";
            this.streaming = true;
            this.ws.onmessage = (event) => {
                this.latestFrame = event.data;

                if (!this.rendering) {
                    this.processFrames();
                }
            };
            this.ws.onclose = (event) => {
                console.log(`Stream closed: code=${event.code}, reason=${event.reason}`);

                this.streaming = false;
                this.latestFrame = null;
                this.ws = null;
            };
            console.log("stream started");
        },

        stop_stream() {
            console.log("stopping stream...");
            this.streaming = false;
            this.latestFrame = null;
            this.ws?.close();
            this.ws = null;
            this.canvas_context.clearRect(0, 0, this.canvas.width, this.canvas.height);
            console.log("stream stopped");
        },

        async processFrames() {
            this.rendering = true;

            try {
                while (this.streaming && this.latestFrame !== null && !this.disposed) {
                    const packet = this.latestFrame;
                    this.latestFrame = null;

                    await this.renderFrame(packet);
                }
            } finally {
                this.rendering = false;
            }
        },

        async renderFrame(buffer) {
            const view = new DataView(buffer);
            const metadataLength = view.getUint32(0, false);

            if (metadataLength > buffer.byteLength - 4) {
                throw new Error("Invalid metadata length");
            }

            const metadata = JSON.parse(this.decoder.decode(new Uint8Array(buffer, 4, metadataLength)));

            const jpegBytes = new Uint8Array(buffer, 4 + metadataLength);

            const bitmap = await createImageBitmap(new Blob([jpegBytes], { type: "image/jpeg" }));

            try {
                if (!this.streaming || this.disposed) return;

                if (this.canvas.width !== bitmap.width || this.canvas.height !== bitmap.height) {
                    this.canvas.width = bitmap.width;
                    this.canvas.height = bitmap.height;
                }

                this.canvas_context.drawImage(bitmap, 0, 0);

                this.canvas_context.strokeStyle = "#00ff66";
                this.canvas_context.fillStyle = "#00ff66";
                this.canvas_context.lineWidth = 2;
                this.canvas_context.font = "16px sans-serif";

                for (const detection of metadata.detections ?? []) {
                    const [x0, y0, x1, y1] = detection.bbox;

                    this.canvas_context.strokeRect(x0, y0, x1 - x0, y1 - y0);

                    const label = `${detection.class} ${detection.depth} ${detection.confidence.toFixed(2)}`;

                    this.canvas_context.fillText(label, x0, Math.max(18, y0 - 6));
                }
            } finally {
                bitmap.close();
            }
        },
    },
};
