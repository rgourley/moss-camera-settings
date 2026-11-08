// Click-to-focus for the Camera Settings node: shows the first render, sets focus_x and focus_y from a click, and outlines what the chosen lens will keep.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SHIFT_Y = 0.05; // matches the headroom Camera Settings gives the crop
const WAITING = "Press Run to see your image here";
const CLICK = "Click your subject, then press Run";
const READY = "Click to move the focus, then press Run";
const CHANGED = "This is a new image. Click your subject, then press Run. If this keeps happening, set Seed to fixed.";

function setup(node) {
    const box = document.createElement("div");
    box.style.cssText = "position:relative;width:100%;height:100%;box-sizing:border-box;display:flex;flex-direction:column;align-items:stretch;justify-content:center;background:#111;border-radius:6px;overflow:hidden;font:13px sans-serif;color:#ddd";
    const stage = document.createElement("div");
    stage.style.cssText = "position:relative;display:none;line-height:0;cursor:crosshair;width:100%;overflow:hidden;flex:none";
    const img = document.createElement("img");
    img.style.cssText = "display:block;width:100%;height:auto";
    img.draggable = false;
    const frame = document.createElement("div");
    frame.style.cssText = "position:absolute;border:2px solid #fff;box-shadow:0 0 0 2000px rgba(0,0,0,.45);pointer-events:none;display:none";
    const mark = document.createElement("div");
    mark.style.cssText = "position:absolute;width:18px;height:18px;margin:-9px 0 0 -9px;border:2px solid #ffd23c;border-radius:50%;box-shadow:0 0 0 1px #000;pointer-events:none;display:none";
    const label = document.createElement("div");
    label.textContent = WAITING;
    label.style.cssText = "padding:6px 8px;line-height:1.3;text-align:center;width:100%;box-sizing:border-box";
    stage.append(img, frame, mark);
    box.append(stage, label);
    const widget = (name) => node.widgets?.find((w) => w.name === name);
    let signature = "", changed = false;
    const record = widget("focus_image");
    if (record) { record.hidden = true; record.computeSize = () => [0, -4]; }

    function draw() {
        const x = widget("focus_x")?.value ?? -1, y = widget("focus_y")?.value ?? -1;
        const picked = x >= 0 && y >= 0;
        if (stage.style.display !== "none") label.textContent = picked ? READY : changed ? CHANGED : CLICK;
        mark.style.display = frame.style.display = picked ? "block" : "none";
        if (!picked) return;
        mark.style.left = `${x * 100}%`;
        mark.style.top = `${y * 100}%`;
        const scale = 35 / (parseInt(widget("lens")?.value) || 35);
        const left = Math.min(Math.max(x - scale / 2, 0), 1 - scale), top = Math.min(Math.max(y + SHIFT_Y - scale / 2, 0), 1 - scale);
        frame.style.left = `${left * 100}%`;
        frame.style.top = `${top * 100}%`;
        frame.style.width = frame.style.height = `${scale * 100}%`;
    }

    stage.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        const r = stage.getBoundingClientRect();
        for (const [name, v] of [["focus_x", (e.clientX - r.left) / r.width], ["focus_y", (e.clientY - r.top) / r.height]]) {
            const w = widget(name);
            if (w) { w.value = Math.round(Math.min(Math.max(v, 0), 1) * 1000) / 1000; w.callback?.(w.value); }
        }
        if (record) { record.value = signature; record.callback?.(signature); }
        changed = false;
        draw();
        app.graph.setDirtyCanvas(true, true);
    });
    const lens = widget("lens");
    if (lens) {
        const cb = lens.callback;
        lens.callback = function (...a) { const out = cb?.apply(this, a); draw(); return out; };
    }
    node.addDOMWidget("focus_picker", "div", box, { serialize: false, getMinHeight: () => 330 });
    node.showPickerImage = (filename, resetFocus, imageSignature) => {
        signature = imageSignature || "";
        changed = !!resetFocus;
        if (resetFocus) {
            if (record) { record.value = ""; record.callback?.(""); }
            for (const name of ["focus_x", "focus_y"]) {
                const w = widget(name);
                if (w) { w.value = -1; w.callback?.(w.value); }
            }
            app.graph.setDirtyCanvas(true, true);
        }
        img.onload = () => { stage.style.display = "block"; draw(); };
        img.src = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=temp&subfolder=&rand=${Date.now()}`);
    };
    node.drawPicker = draw;
}

app.registerExtension({
    name: "depth_lens_fx.camera_settings",
    async setup() {
        api.addEventListener("depth_lens_fx.picker", ({ detail }) => app.graph?.getNodeById(Number(detail.node))?.showPickerImage?.(detail.filename, detail.reset_focus, detail.signature));
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CameraSettings") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () { const out = created?.apply(this, arguments); setup(this); return out; };
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const out = configure?.apply(this, arguments);
            // a workflow saved before this menu existed has no value for it
            const edge = this.widgets?.find((w) => w.name === "subject_edge");
            if (edge && !edge.options?.values?.includes(edge.value)) edge.value = "Natural";
            const lens = this.widgets?.find((w) => w.name === "lens");
            if (lens && !lens.options?.values?.includes(lens.value)) lens.value = "85mm";
            const front = this.widgets?.find((w) => w.name === "foreground");
            if (front && !front.options?.values?.includes(front.value)) front.value = "Sharp";
            const nudge = this.widgets?.find((w) => w.name === "focus_nudge");
            if (nudge && !nudge.options?.values?.includes(nudge.value)) nudge.value = "As clicked";
            this.drawPicker?.();
            return out;
        };
    },
});
