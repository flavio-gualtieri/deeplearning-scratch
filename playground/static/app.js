const state = {
  csv: { filename: "", columns: [], columnTypes: {} }, // columnTypes: { column: "numeric" | "categorical" }
  target: null,
  excludedFeatures: new Set(), // columns unchecked in the left panel
  encoderGroups: [], // [{ id, type, embedding_dim, hidden_dims, dropout, featureColumns: [] }]
  head: null, // { type, output_dim?, hidden_dims, dropout }
  training: {
    optimizer: "adam",
    loss_fn: "cross_entropy",
    splits: [0.7, 0.15, 0.15],
    epochs: 50,
    batch_size: 16,
    learning_rate: 0.001,
    device: "",
  },
};

const DEFAULTS = {
  mlp: { embedding_dim: 16, hidden_dims: "32", dropout: 0 },
  categorical: { embedding_dim: 8, dropout: 0 },
  cnn: { embedding_dim: 16, channels: "32, 64, 128", kernel_size: 3, dropout: 0 },
  rnn: { embedding_dim: 16, hidden_dim: 128, num_layers: 1, bidirectional: false, dropout: 0 },
  transformer: { embedding_dim: 16, model_dim: 128, num_heads: 4, num_layers: 2, feedforward_dim: 256, dropout: 0 },
  classification: { hidden_dims: "", dropout: 0 },
  regression: { output_dim: 1, hidden_dims: "", dropout: 0 },
  multilabel: { output_dim: 2, hidden_dims: "", dropout: 0 },
  projection: { output_dim: 128, hidden_dims: "", normalize: true },
  sequence_tagging: { output_dim: 2, hidden_dims: "", dropout: 0 },
};

const ENCODER_LABELS = {
  mlp: "MLP Encoder",
  categorical: "Categorical Encoder",
  cnn: "Image CNN Encoder",
  rnn: "Sequence RNN Encoder",
  transformer: "Sequence Transformer Encoder",
};

const HEAD_LABELS = {
  classification: "Classification Head",
  regression: "Regression Head",
  multilabel: "Multi-label Head",
  projection: "Projection Head",
  sequence_tagging: "Sequence Tagging Head",
};

// Mirrors each leaf encoder's `accepted_feature_types` on the Python side
// (see deepscratch/encoders/*.py) so a mismatched drop is rejected here too,
// not just after the training subprocess fails. cnn/rnn/transformer accept
// "numeric" structurally (a MultiEncoder group is always plain float columns)
// even though they really want image/sequence data -- see the shape guards
// in their forward() methods for what actually happens if you run one.
const ACCEPTED_FEATURE_TYPES = {
  mlp: ["numeric"],
  categorical: ["categorical"],
  cnn: ["numeric"],
  rnn: ["numeric"],
  transformer: ["numeric"],
};

let chartPoints = [];
let pollTimer = null;
let nextGroupId = 1;

function el(id) {
  return document.getElementById(id);
}

// Display-only: turns a raw CSV column name into a friendly label
// ("sepal_length" -> "Sepal Length"). The raw name is always what's kept in
// state and sent to the backend -- only text shown to the user goes through this.
function formatColumnLabel(column) {
  return column
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

async function init() {
  const response = await fetch("/api/csv");
  const data = await response.json();

  state.csv.filename = data.filename;
  state.csv.columns = data.columns;
  state.csv.columnTypes = data.column_types || {};

  state.encoderGroups = [{ id: nextGroupId++, type: null, featureColumns: [] }];

  renderColumnStack();
  renderEncoderGroups();
  wireSlots();
  wirePalette();
  wireBottomBar();

  el("add-encoder-group").addEventListener("click", () => {
    state.encoderGroups.push({ id: nextGroupId++, type: null, featureColumns: [] });
    renderEncoderGroups();
  });

  el("run-button").addEventListener("click", onRun);
}

// ---- rendering ----

function renderColumnStack() {
  el("csv-name").textContent = state.csv.filename;

  const stack = el("column-stack");
  stack.innerHTML = "";

  const assignedColumns = new Set(state.encoderGroups.flatMap((g) => g.featureColumns));

  for (const column of state.csv.columns) {
    if (column === state.target) continue;
    if (assignedColumns.has(column)) continue; // moved into an encoder group -- shown there instead

    const included = !state.excludedFeatures.has(column);
    const columnType = state.csv.columnTypes[column] || "numeric";

    const chip = document.createElement("div");
    chip.className = ["block", "column-chip", `type-${columnType}`, included ? "" : "excluded"]
      .filter(Boolean)
      .join(" ");
    chip.draggable = true;
    chip.title = columnType;
    chip.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", JSON.stringify({ kind: "column", column }));
    });

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = included;
    checkbox.title = "Include this column as a feature";
    checkbox.addEventListener("click", (event) => event.stopPropagation());
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        state.excludedFeatures.delete(column);
      } else {
        state.excludedFeatures.add(column);
      }
      renderColumnStack();
    });

    const label = document.createElement("span");
    label.textContent = formatColumnLabel(column);

    chip.appendChild(checkbox);
    chip.appendChild(label);
    stack.appendChild(chip);
  }
}

function renderTargetSlot() {
  const slot = el("target-slot");
  slot.classList.toggle("filled", Boolean(state.target));

  if (!state.target) {
    slot.innerHTML = "Drop Target Column Here";
    return;
  }

  slot.innerHTML = `<button class="remove-btn" title="Remove target">&times;</button>Target: ${formatColumnLabel(state.target)}`;
  slot.querySelector(".remove-btn").addEventListener("click", () => {
    state.target = null;
    renderTargetSlot();
    renderColumnStack();
  });
}

function renderEncoderFields(group) {
  if (group.type === "mlp") {
    return `
      <label>Embedding dim <input type="number" min="1" data-field="embedding_dim" value="${group.embedding_dim}"></label>
      <label>Hidden dims <input type="text" data-field="hidden_dims" value="${group.hidden_dims}"></label>
      <label>Dropout <input type="number" min="0" max="0.9" step="0.05" data-field="dropout" value="${group.dropout}"></label>
    `;
  }

  if (group.type === "categorical") {
    return `
      <label>Embedding dim <input type="number" min="1" data-field="embedding_dim" value="${group.embedding_dim}"></label>
      <label>Dropout <input type="number" min="0" max="0.9" step="0.05" data-field="dropout" value="${group.dropout}"></label>
    `;
  }

  if (group.type === "cnn") {
    return `
      <label>Embedding dim <input type="number" min="1" data-field="embedding_dim" value="${group.embedding_dim}"></label>
      <label>Channels <input type="text" data-field="channels" value="${group.channels}"></label>
      <label>Kernel size <input type="number" min="1" step="2" data-field="kernel_size" value="${group.kernel_size}"></label>
      <label>Dropout <input type="number" min="0" max="0.9" step="0.05" data-field="dropout" value="${group.dropout}"></label>
    `;
  }

  if (group.type === "rnn") {
    return `
      <label>Embedding dim <input type="number" min="1" data-field="embedding_dim" value="${group.embedding_dim}"></label>
      <label>Hidden dim <input type="number" min="1" data-field="hidden_dim" value="${group.hidden_dim}"></label>
      <label>Layers <input type="number" min="1" data-field="num_layers" value="${group.num_layers}"></label>
      <label class="checkbox-label"><input type="checkbox" data-field="bidirectional" ${group.bidirectional ? "checked" : ""}> Bidirectional</label>
      <label>Dropout <input type="number" min="0" max="0.9" step="0.05" data-field="dropout" value="${group.dropout}"></label>
    `;
  }

  if (group.type === "transformer") {
    return `
      <label>Embedding dim <input type="number" min="1" data-field="embedding_dim" value="${group.embedding_dim}"></label>
      <label>Model dim <input type="number" min="1" data-field="model_dim" value="${group.model_dim}"></label>
      <label>Heads <input type="number" min="1" data-field="num_heads" value="${group.num_heads}"></label>
      <label>Layers <input type="number" min="1" data-field="num_layers" value="${group.num_layers}"></label>
      <label>Feedforward dim <input type="number" min="1" data-field="feedforward_dim" value="${group.feedforward_dim}"></label>
      <label>Dropout <input type="number" min="0" max="0.9" step="0.05" data-field="dropout" value="${group.dropout}"></label>
    `;
  }

  return "";
}

function renderEncoderGroups() {
  const container = el("encoder-groups");
  container.innerHTML = "";

  for (const group of state.encoderGroups) {
    const groupEl = document.createElement("div");
    groupEl.className = "slot encoder-group" + (group.type ? " filled type-" + group.type : "");

    const chipsHtml = group.featureColumns
      .map(
        (column) =>
          `<span class="assigned-chip">${formatColumnLabel(column)}<button class="chip-remove" data-column="${column}" title="Unassign">&times;</button></span>`
      )
      .join("");

    groupEl.innerHTML = group.type
      ? `
        <button class="remove-btn" title="Remove group">&times;</button>
        <div class="block-title">${ENCODER_LABELS[group.type]}</div>
        ${renderEncoderFields(group)}
        <div class="feature-drop-zone">
          <div class="feature-drop-label">Features</div>
          ${chipsHtml || '<span class="drop-hint-small">Drag columns here</span>'}
        </div>
      `
      : `
        <button class="remove-btn" title="Remove group">&times;</button>
        <div class="drop-hint">Drop Encoder Here</div>
      `;

    container.appendChild(groupEl);
    wireEncoderGroupEvents(groupEl, group);
  }
}

function renderHeadSlot() {
  const slot = el("head-slot");

  if (!state.head) {
    slot.classList.remove("filled");
    slot.innerHTML = "Drop Head Here";
    return;
  }

  slot.classList.add("filled");

  const needsOutputDim = ["regression", "multilabel", "projection", "sequence_tagging"].includes(
    state.head.type
  );
  const outputDimLabel = { regression: "Output dim", multilabel: "Num labels", projection: "Projection dim", sequence_tagging: "Num tags" }[
    state.head.type
  ];
  const outputDimField = needsOutputDim
    ? `<label>${outputDimLabel} <input type="number" min="1" data-field="output_dim" value="${state.head.output_dim}"></label>`
    : "";

  const dropoutField =
    state.head.type === "projection"
      ? `<label class="checkbox-label"><input type="checkbox" data-field="normalize" ${state.head.normalize ? "checked" : ""}> L2-normalize output</label>`
      : `<label>Dropout <input type="number" min="0" max="0.9" step="0.05" data-field="dropout" value="${state.head.dropout}"></label>`;

  slot.innerHTML = `
    <button class="remove-btn" title="Remove head">&times;</button>
    <div class="block-title">${HEAD_LABELS[state.head.type]}</div>
    ${outputDimField}
    <label>Hidden dims <input type="text" data-field="hidden_dims" value="${state.head.hidden_dims}"></label>
    ${dropoutField}
  `;

  wireSlotRemove(slot, () => {
    state.head = null;
    renderHeadSlot();
  });
  wireSlotInputs(slot, state.head);
}

function wireSlotRemove(slot, onRemove) {
  slot.querySelector(".remove-btn").addEventListener("click", (event) => {
    event.stopPropagation();
    onRemove();
  });
}

function wireSlotInputs(slot, target) {
  slot.querySelectorAll("input[data-field]").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.type === "checkbox") {
        target[input.dataset.field] = input.checked;
      } else if (input.type === "number") {
        target[input.dataset.field] = Number(input.value);
      } else {
        target[input.dataset.field] = input.value;
      }
    });
  });
}

// ---- drag and drop ----

function wirePalette() {
  document.querySelectorAll(".palette-block").forEach((block) => {
    block.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData(
        "text/plain",
        JSON.stringify({ kind: "component", role: block.dataset.role, type: block.dataset.type })
      );
    });
  });
}

function wireSlots() {
  const slots = [
    { node: el("head-slot"), role: "head" },
    { node: el("target-slot"), role: "target" },
  ];

  for (const { node, role } of slots) {
    node.addEventListener("dragover", (event) => {
      event.preventDefault();
      node.classList.add("dragover");
    });
    node.addEventListener("dragleave", () => node.classList.remove("dragover"));
    node.addEventListener("drop", (event) => {
      event.preventDefault();
      node.classList.remove("dragover");
      handleDrop(role, event);
    });
  }
}

function handleDrop(role, event) {
  const payload = JSON.parse(event.dataTransfer.getData("text/plain") || "{}");

  if (role === "target") {
    if (payload.kind !== "column") return;
    state.excludedFeatures.delete(payload.column);
    for (const group of state.encoderGroups) {
      group.featureColumns = group.featureColumns.filter((c) => c !== payload.column);
    }
    state.target = payload.column;
    renderTargetSlot();
    renderColumnStack();
    renderEncoderGroups();
    return;
  }

  if (payload.kind !== "component" || payload.role !== role) {
    setStatus("That component doesn't belong in this slot.");
    return;
  }

  if (role === "head") {
    state.head = { type: payload.type, ...DEFAULTS[payload.type] };
    renderHeadSlot();
  }
}

function wireEncoderGroupEvents(groupEl, group) {
  wireSlotRemove(groupEl, () => {
    state.encoderGroups = state.encoderGroups.filter((g) => g.id !== group.id);
    if (state.encoderGroups.length === 0) {
      state.encoderGroups.push({ id: nextGroupId++, type: null, featureColumns: [] });
    }
    renderEncoderGroups();
    renderColumnStack();
  });

  groupEl.addEventListener("dragover", (event) => {
    event.preventDefault();
    groupEl.classList.add("dragover");
  });
  groupEl.addEventListener("dragleave", () => groupEl.classList.remove("dragover"));
  groupEl.addEventListener("drop", (event) => {
    event.preventDefault();
    groupEl.classList.remove("dragover");
    handleEncoderGroupDrop(group, event);
  });

  if (group.type) {
    wireSlotInputs(groupEl, group);

    groupEl.querySelectorAll(".chip-remove").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        const column = button.dataset.column;
        group.featureColumns = group.featureColumns.filter((c) => c !== column);
        renderEncoderGroups();
        renderColumnStack();
      });
    });
  }
}

function handleEncoderGroupDrop(group, event) {
  const payload = JSON.parse(event.dataTransfer.getData("text/plain") || "{}");

  if (payload.kind === "component") {
    if (payload.role !== "encoder") {
      setStatus("That component doesn't belong in an encoder group.");
      return;
    }

    if (group.type) {
      setStatus("This group already has an encoder — remove it first to change the type.");
      return;
    }

    Object.assign(group, { type: payload.type, ...DEFAULTS[payload.type] });
    renderEncoderGroups();
    return;
  }

  if (payload.kind === "column") {
    if (!group.type) {
      setStatus("Drop an encoder into this group before assigning feature columns.");
      return;
    }

    if (payload.column === state.target) return;

    const columnType = state.csv.columnTypes[payload.column] || "numeric";
    const accepted = ACCEPTED_FEATURE_TYPES[group.type];

    if (accepted && !accepted.includes(columnType)) {
      setStatus(
        `"${formatColumnLabel(payload.column)}" is ${columnType}, but ${ENCODER_LABELS[group.type]} only accepts ${accepted.join(", ")} columns.`
      );
      return;
    }

    for (const other of state.encoderGroups) {
      other.featureColumns = other.featureColumns.filter((c) => c !== payload.column);
    }

    group.featureColumns.push(payload.column);
    state.excludedFeatures.delete(payload.column);

    renderEncoderGroups();
    renderColumnStack();
  }
}

// ---- bottom bar ----

function wireBottomBar() {
  el("optimizer").addEventListener("change", (e) => (state.training.optimizer = e.target.value));
  el("loss_fn").addEventListener("change", (e) => (state.training.loss_fn = e.target.value));
  el("epochs").addEventListener("change", (e) => (state.training.epochs = Number(e.target.value)));
  el("batch_size").addEventListener("change", (e) => (state.training.batch_size = Number(e.target.value)));
  el("learning_rate").addEventListener("change", (e) => (state.training.learning_rate = Number(e.target.value)));
  el("device").addEventListener("change", (e) => (state.training.device = e.target.value));

  ["split-train", "split-val", "split-test"].forEach((id) => {
    el(id).addEventListener("change", () => {
      state.training.splits = [
        Number(el("split-train").value),
        Number(el("split-val").value),
        Number(el("split-test").value),
      ];
    });
  });
}

// ---- run + polling ----

function setStatus(text) {
  el("status-text").textContent = text;
}

function parseDims(text) {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .map(Number)
    .filter((value) => Number.isInteger(value) && value > 0);
}

// Builds the `encoder` object for one group's /api/run payload. Each encoder
// type only cares about a subset of the group's fields -- pulling exactly
// those (rather than sending the whole group) keeps config.yaml free of
// leftover fields that don't apply to that type.
function buildEncoderPayload(group) {
  const base = { type: group.type, embedding_dim: group.embedding_dim, dropout: group.dropout };

  if (group.type === "mlp") {
    return { ...base, hidden_dims: parseDims(group.hidden_dims || "") };
  }

  if (group.type === "categorical") {
    return base;
  }

  if (group.type === "cnn") {
    return { ...base, channels: parseDims(group.channels || ""), kernel_size: group.kernel_size };
  }

  if (group.type === "rnn") {
    return {
      ...base,
      hidden_dim: group.hidden_dim,
      num_layers: group.num_layers,
      bidirectional: Boolean(group.bidirectional),
    };
  }

  if (group.type === "transformer") {
    return {
      ...base,
      model_dim: group.model_dim,
      num_heads: group.num_heads,
      num_layers: group.num_layers,
      feedforward_dim: group.feedforward_dim,
    };
  }

  return base;
}

async function onRun() {
  const filledGroups = state.encoderGroups.filter((g) => g.type);

  if (filledGroups.length === 0) {
    return setStatus("Drop at least one encoder into an encoder group.");
  }

  if (!state.head) return setStatus("Add a head first.");
  if (!state.target) return setStatus("Choose a target column first.");

  const includedColumns = state.csv.columns.filter(
    (column) => column !== state.target && !state.excludedFeatures.has(column)
  );

  if (includedColumns.length === 0) {
    return setStatus("Select at least one feature column (checkboxes on the left).");
  }

  const assignedColumns = new Set(filledGroups.flatMap((g) => g.featureColumns));
  const unassigned = includedColumns.filter((column) => !assignedColumns.has(column));

  if (unassigned.length > 0) {
    return setStatus(
      `Drag these columns into an encoder group first: ${unassigned.map(formatColumnLabel).join(", ")}`
    );
  }

  const payload = {
    target: state.target,
    encoder_groups: filledGroups.map((group) => ({
      feature_columns: group.featureColumns,
      encoder: buildEncoderPayload(group),
    })),
    head: { ...state.head, hidden_dims: parseDims(state.head.hidden_dims) },
    training: state.training,
  };

  el("run-button").disabled = true;
  chartPoints = [];
  drawChart();
  setStatus("starting…");

  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const body = await response.json();
    setStatus(`Error: ${body.error}`);
    el("run-button").disabled = false;
    return;
  }

  poll();
}

function poll() {
  clearTimeout(pollTimer);

  pollTimer = setTimeout(async () => {
    const response = await fetch("/api/status");
    const data = await response.json();

    chartPoints = data.points;
    drawChart();

    if (data.status === "running") {
      const last = chartPoints[chartPoints.length - 1];
      setStatus(last ? `Epoch ${last.epoch}/${last.total}` : "running…");
      poll();
    } else if (data.status === "done") {
      setStatus("Done.");
      el("run-button").disabled = false;
    } else if (data.status === "error") {
      setStatus(`Error:\n${data.error}`);
      el("run-button").disabled = false;
    }
  }, 600);
}

// ---- chart ----

function drawChart() {
  const canvas = el("loss-chart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (chartPoints.length === 0) return;

  const values = chartPoints.flatMap((p) => [p.train_loss, p.val_loss].filter((v) => v != null));
  const maxValue = Math.max(...values, 1e-6);
  const minValue = Math.min(...values, 0);
  const maxEpoch = Math.max(chartPoints[chartPoints.length - 1].epoch, 1);

  const xScale = (epoch) => 10 + (epoch / maxEpoch) * (canvas.width - 20);
  const yScale = (value) =>
    canvas.height - 10 - ((value - minValue) / (maxValue - minValue || 1)) * (canvas.height - 20);

  drawLine(ctx, "train_loss", "#4f8cff", xScale, yScale);
  drawLine(ctx, "val_loss", "#ff6b6b", xScale, yScale);
}

function drawLine(ctx, key, color, xScale, yScale) {
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  let started = false;

  for (const point of chartPoints) {
    const value = point[key];
    if (value == null) continue;

    const x = xScale(point.epoch);
    const y = yScale(value);

    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
}

init();
