type VideoValidation = {
  is_valid: boolean;
  duration_seconds?: number;
  fps?: number;
  frame_count?: number;
  width?: number;
  height?: number;
  size_mb?: number;
  extension?: string;
  source?: "browser" | "backend";
};

type FrameScore = {
  frame: number;
  fake_score: number;
};

type ModelMetadata = {
  architecture: string;
  checkpoint_path: string;
  device: string;
  model_loaded: boolean;
  load_error?: string | null;
  fake_class_index: number;
  normalization: string;
  label_map: string[];
  num_classes: number;
};

type ReportInput = {
  fileName: string;
  prediction: "real" | "fake" | "uncertain";
  confidence: number | null;
  fakeScore: number | null;
  decisionThreshold: number | null;
  analysisStartTime: number | null;
  analyzedSegments: number[];
  frameScores: FrameScore[];
  framesCount: number;
  heatmapsCount: number;
  videoValidation: VideoValidation | null;
  modelMetadata: ModelMetadata | null;
};

const formatTime = (totalSeconds: number) => {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
};

const escapePdfText = (value: string) =>
  value.replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)");

const wrapText = (text: string, maxChars = 82) => {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length <= maxChars) {
      current = next;
    } else {
      if (current) lines.push(current);
      current = word;
    }
  }

  if (current) lines.push(current);
  return lines;
};

const buildLines = (input: ReportInput) => {
  const now = new Date();
  const predictionLabel =
    input.prediction === "real"
      ? "Authentic Video"
      : input.prediction === "fake"
        ? "Deepfake Detected"
        : "Needs Review";

  const lines: string[] = [
    "Deepfake Detection Analysis Report",
    "",
    `Generated: ${now.toLocaleString()}`,
    `Video File: ${input.fileName}`,
    `Prediction: ${predictionLabel}`,
  ];

  if (input.confidence !== null) {
    lines.push(`Confidence: ${(input.confidence * 100).toFixed(2)}%`);
  }
  if (input.fakeScore !== null) {
    lines.push(`Fake Score: ${(input.fakeScore * 100).toFixed(2)}%`);
  }
  if (input.decisionThreshold !== null) {
    lines.push(`Decision Threshold: ${(input.decisionThreshold * 100).toFixed(2)}%`);
  }
  if (input.analysisStartTime !== null) {
    lines.push(`Analysis Start Time: ${formatTime(input.analysisStartTime)}`);
  }

  lines.push(`Extracted Frames: ${input.framesCount}`);
  lines.push(`Grad-CAM Heatmaps: ${input.heatmapsCount}`);
  lines.push("");
  lines.push("Analyzed Segments");

  if (input.analyzedSegments.length > 0) {
    for (const segment of input.analyzedSegments) {
      lines.push(`- ${formatTime(segment)}`);
    }
  } else {
    lines.push("- None");
  }

  lines.push("");
  lines.push("Video Validity Check");
  if (input.videoValidation) {
    lines.push(`- Status: ${input.videoValidation.is_valid ? "Valid" : "Invalid"}`);
    if (input.videoValidation.extension) {
      lines.push(`- Extension: ${input.videoValidation.extension.toUpperCase()}`);
    }
    if (input.videoValidation.size_mb !== undefined) {
      lines.push(`- Size: ${input.videoValidation.size_mb.toFixed(2)} MB`);
    }
    if (input.videoValidation.duration_seconds !== undefined) {
      lines.push(`- Duration: ${formatTime(input.videoValidation.duration_seconds)}`);
    }
    if (input.videoValidation.width && input.videoValidation.height) {
      lines.push(`- Resolution: ${input.videoValidation.width}x${input.videoValidation.height}`);
    }
    if (input.videoValidation.fps !== undefined) {
      lines.push(`- FPS: ${input.videoValidation.fps.toFixed(2)}`);
    }
    if (input.videoValidation.frame_count !== undefined) {
      lines.push(`- Frame Count: ${input.videoValidation.frame_count}`);
    }
  } else {
    lines.push("- Not available");
  }

  lines.push("");
  lines.push("Frame-Level Fake Scores");
  if (input.frameScores.length > 0) {
    for (const item of input.frameScores) {
      lines.push(`- Frame ${item.frame}: ${(item.fake_score * 100).toFixed(2)}%`);
    }
  } else {
    lines.push("- Not available");
  }

  lines.push("");
  lines.push("Model Metadata");
  if (input.modelMetadata) {
    lines.push(`- Architecture: ${input.modelMetadata.architecture}`);
    lines.push(`- Device: ${input.modelMetadata.device}`);
    lines.push(`- Model Status: ${input.modelMetadata.model_loaded ? "Loaded" : "Not loaded"}`);
    lines.push(`- Checkpoint: ${input.modelMetadata.checkpoint_path}`);
    lines.push(`- Normalization: ${input.modelMetadata.normalization}`);
    lines.push(`- Fake Class Index: ${input.modelMetadata.fake_class_index}`);
    lines.push(`- Classes: ${input.modelMetadata.num_classes}`);
    lines.push(`- Label Map: ${input.modelMetadata.label_map.join(", ")}`);
    if (input.modelMetadata.load_error) {
      lines.push(`- Load Error: ${input.modelMetadata.load_error}`);
    }
  } else {
    lines.push("- Not available");
  }

  return lines.flatMap((line) => wrapText(line));
};

const createPdfBlob = (lines: string[]) => {
  const pageWidth = 595;
  const pageHeight = 842;
  const marginLeft = 50;
  const topStart = 790;
  const lineHeight = 16;
  const linesPerPage = 45;
  const pages: string[][] = [];

  for (let index = 0; index < lines.length; index += linesPerPage) {
    pages.push(lines.slice(index, index + linesPerPage));
  }

  let objectIndex = 1;
  const objects: string[] = [];
  const pageObjectIds: number[] = [];

  const fontId = objectIndex++;
  objects[fontId] = `${fontId} 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
`;

  const pagesId = objectIndex++;

  for (const pageLines of pages) {
    const contentId = objectIndex++;
    const pageId = objectIndex++;
    pageObjectIds.push(pageId);

    let contentStream = "BT\n/F1 12 Tf\n";
    pageLines.forEach((line, index) => {
      const y = topStart - index * lineHeight;
      contentStream += `1 0 0 1 ${marginLeft} ${y} Tm (${escapePdfText(line)}) Tj\n`;
    });
    contentStream += "ET";

    objects[contentId] = `${contentId} 0 obj
<< /Length ${contentStream.length} >>
stream
${contentStream}
endstream
endobj
`;

    objects[pageId] = `${pageId} 0 obj
<< /Type /Page /Parent ${pagesId} 0 R /MediaBox [0 0 ${pageWidth} ${pageHeight}] /Resources << /Font << /F1 ${fontId} 0 R >> >> /Contents ${contentId} 0 R >>
endobj
`;
  }

  const kids = pageObjectIds.map((id) => `${id} 0 R`).join(" ");
  objects[pagesId] = `${pagesId} 0 obj
<< /Type /Pages /Kids [${kids}] /Count ${pageObjectIds.length} >>
endobj
`;

  const catalogId = objectIndex++;
  objects[catalogId] = `${catalogId} 0 obj
<< /Type /Catalog /Pages ${pagesId} 0 R >>
endobj
`;

  let pdf = "%PDF-1.4\n";
  const offsets: number[] = [0];

  for (let index = 1; index < objectIndex; index++) {
    offsets[index] = pdf.length;
    pdf += objects[index];
  }

  const xrefOffset = pdf.length;
  pdf += `xref
0 ${objectIndex}
0000000000 65535 f 
`;

  for (let index = 1; index < objectIndex; index++) {
    pdf += `${String(offsets[index]).padStart(10, "0")} 00000 n 
`;
  }

  pdf += `trailer
<< /Size ${objectIndex} /Root ${catalogId} 0 R >>
startxref
${xrefOffset}
%%EOF`;

  return new Blob([pdf], { type: "application/pdf" });
};

export const downloadAnalysisPdf = (input: ReportInput) => {
  const blob = createPdfBlob(buildLines(input));
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const baseName = input.fileName.replace(/\.[^/.]+$/, "") || "analysis-report";

  anchor.href = url;
  anchor.download = `${baseName}-analysis-report.pdf`;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
};
