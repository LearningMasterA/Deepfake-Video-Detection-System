import { useState } from "react";
import { Upload, FileVideo, AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { analyzeVideoAPI } from "@/lib/api";

type FrameScore = {
  frame: number;
  fake_score: number;
};

export const UploadSection = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<"real" | "fake" | null>(null);
  const [frames, setFrames] = useState<string[]>([]);
  const [confidence, setConfidence] = useState<number | null>(null);
  const { toast } = useToast();
  const [heatmaps, setHeatmaps] = useState<string[]>([]);
  const [startMinute, setStartMinute] = useState("0");
  const [startSecond, setStartSecond] = useState("0");
  const [fakeScore, setFakeScore] = useState<number | null>(null);
  const [decisionThreshold, setDecisionThreshold] = useState<number | null>(null);
  const [analysisStartTime, setAnalysisStartTime] = useState<number | null>(null);
  const [frameScores, setFrameScores] = useState<FrameScore[]>([]);

  const formatTime = (totalSeconds: number) => {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);

    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("video/")) {
      setFile(droppedFile);
      setResult(null);
      setFrames([]);
      setHeatmaps([]);
      setConfidence(null);
      setFakeScore(null);
      setDecisionThreshold(null);
      setAnalysisStartTime(null);
      setFrameScores([]);
    } else {
      toast({
        title: "Invalid file type",
        description: "Please upload a video file",
        variant: "destructive",
      });
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setFrames([]);
      setConfidence(null);
      setHeatmaps([]);
      setFakeScore(null);
      setDecisionThreshold(null);
      setAnalysisStartTime(null);
      setFrameScores([]);
    }
  };

  const analyzeVideo = async () => {
  if (!file) return;

  setIsAnalyzing(true);
  setResult(null);
  setFrames([]);
  setHeatmaps([]);
  setConfidence(null);
  setFakeScore(null);
  setDecisionThreshold(null);
  setAnalysisStartTime(null);
  setFrameScores([]);

  try {
    const minutes = Math.max(0, Number(startMinute) || 0);
    const seconds = Math.min(59, Math.max(0, Number(startSecond) || 0));
    const startTime = minutes * 60 + seconds;
    const data = await analyzeVideoAPI(file, startTime);

    setResult(data.prediction === "Real" ? "real" : "fake");
    setFrames(data.frames || []);
    setHeatmaps(data.heatmaps || []);
    setConfidence(data.confidence);
    setFakeScore(data.fake_score ?? null);
    setDecisionThreshold(data.decision_threshold ?? null);
    setAnalysisStartTime(data.start_time ?? startTime);
    setFrameScores(data.frame_scores || []);

    toast({
      title: "Analysis Complete",
      description: `Confidence: ${data.confidence}`,
    });

  } catch (error) {
    toast({
      title: "Error",
      description: "Failed to analyze video. Backend not reachable.",
      variant: "destructive",
    });
  } finally {
    setIsAnalyzing(false);
  }
};

  return (
    <section id="upload-section" className="py-24 px-4 bg-background/50">
      <div className="container mx-auto max-w-4xl">
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Analyze Your Video
          </h2>
          <p className="text-muted-foreground text-lg">
            Upload a video file to detect deepfake manipulation
          </p>
        </div>

        <Card className="bg-card shadow-card border-border/50">
          <CardContent className="p-8">
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
                isDragging
                  ? "border-primary bg-primary/5 scale-105"
                  : "border-border hover:border-primary/50"
              }`}
            >
              {!file ? (
                <>
                  <Upload className="w-16 h-16 mx-auto mb-4 text-primary" />
                  <h3 className="text-xl font-semibold mb-2">
                    Drop your video here
                  </h3>
                  <p className="text-muted-foreground mb-6">
                    or click to browse files
                  </p>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileInput}
                    className="hidden"
                    id="video-upload"
                  />
                  <label htmlFor="video-upload">
                    <Button variant="outline" className="cursor-pointer" asChild>
                      <span>
                        <FileVideo className="mr-2 h-4 w-4" />
                        Select Video
                      </span>
                    </Button>
                  </label>
                </>
              ) : (
                <>
                  <FileVideo className="w-16 h-16 mx-auto mb-4 text-primary" />
                  <h3 className="text-xl font-semibold mb-2">{file.name}</h3>
                  <p className="text-muted-foreground mb-6">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                  {!result && !isAnalyzing && (
                    <div className="mx-auto mb-6 max-w-xs rounded-xl border border-border bg-background/60 p-4 text-left">
                      <label className="mb-3 block text-sm font-medium">
                        Start analysis from
                      </label>
                      <div className="flex items-center gap-3">
                        <div className="flex-1">
                          <span className="mb-1 block text-xs text-muted-foreground">
                            Minutes
                          </span>
                          <input
                            type="number"
                            min="0"
                            value={startMinute}
                            onChange={(e) => setStartMinute(e.target.value)}
                            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                          />
                        </div>
                        <span className="pt-5 text-muted-foreground">:</span>
                        <div className="flex-1">
                          <span className="mb-1 block text-xs text-muted-foreground">
                            Seconds
                          </span>
                          <input
                            type="number"
                            min="0"
                            max="59"
                            value={startSecond}
                            onChange={(e) => setStartSecond(e.target.value)}
                            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                          />
                        </div>
                      </div>
                      <p className="mt-2 text-xs text-muted-foreground">
                        Frames will be extracted after this timestamp.
                      </p>
                    </div>
                  )}
                  {!result && !isAnalyzing && (
                    <Button
                      onClick={analyzeVideo}
                      className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow"
                    >
                      <Brain className="mr-2 h-4 w-4" />
                      Analyze Video
                    </Button>
                  )}
                  {isAnalyzing && (
                    <div className="mt-6 flex flex-col items-center justify-center gap-3 rounded-xl border border-primary/20 bg-primary/5 p-6">
                      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                        <Loader2 className="h-9 w-9 animate-spin text-primary" />
                      </div>
                      <div className="text-center">
                        <p className="font-semibold">Analyzing video...</p>
                        <p className="text-sm text-muted-foreground">
                          Extracting face frames and generating Grad-CAM heatmaps.
                        </p>
                      </div>
                    </div>
                  )}
                  {result && confidence !== null && (
  <div
    className={`mt-8 p-8 rounded-2xl shadow-xl transition-all duration-500 animate-in fade-in zoom-in-95 ${
      result === "real"
        ? "bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border border-emerald-500/30"
        : "bg-gradient-to-br from-red-500/10 to-red-600/5 border border-red-500/30"
    }`}
  >
    <div className="flex flex-col items-center text-center">
      <div
        className={`w-16 h-16 flex items-center justify-center rounded-full mb-4 ${
          result === "real"
            ? "bg-emerald-500/20"
            : "bg-red-500/20"
        }`}
      >
        {result === "real" ? (
          <CheckCircle2 className="w-10 h-10 text-emerald-500" />
        ) : (
          <AlertCircle className="w-10 h-10 text-red-500" />
        )}
      </div>

      <h3 className="text-3xl font-bold mb-2">
        {result === "real" ? "Authentic Video" : "Deepfake Detected"}
      </h3>

      <p className="text-muted-foreground mb-6 max-w-md">
        {result === "real"
          ? "No significant manipulation patterns were detected."
          : "Model detected suspicious manipulation patterns in the facial region."}
      </p>

      {/* Confidence Progress Bar */}
      <div className="w-full max-w-md">
        <div className="flex justify-between text-sm mb-2">
          <span>Confidence Score</span>
          <span className="font-semibold">
            {(confidence * 100).toFixed(2)}%
          </span>
        </div>

        <div className="w-full h-3 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-700 ${
              result === "real"
                ? "bg-emerald-500"
                : "bg-red-500"
            }`}
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      <div className="mt-6 grid w-full max-w-md grid-cols-1 gap-3 text-sm sm:grid-cols-3">
        {fakeScore !== null && (
          <div className="rounded-lg border border-border bg-background/60 p-3">
            <p className="text-muted-foreground">Fake Score</p>
            <p className="font-semibold">{(fakeScore * 100).toFixed(2)}%</p>
          </div>
        )}
        {decisionThreshold !== null && (
          <div className="rounded-lg border border-border bg-background/60 p-3">
            <p className="text-muted-foreground">Threshold</p>
            <p className="font-semibold">{(decisionThreshold * 100).toFixed(0)}%</p>
          </div>
        )}
        {analysisStartTime !== null && (
          <div className="rounded-lg border border-border bg-background/60 p-3">
            <p className="text-muted-foreground">Started At</p>
            <p className="font-semibold">{formatTime(analysisStartTime)}</p>
          </div>
        )}
      </div>
    </div>
  </div>
)}
                  <Button
                    variant="ghost"
                    className="mt-4"
                    onClick={() => {
                      setFile(null);
                      setResult(null);
                      setFrames([]);
                      setHeatmaps([]);
                      setConfidence(null);
                      setFakeScore(null);
                      setDecisionThreshold(null);
                      setAnalysisStartTime(null);
                      setFrameScores([]);
                      setStartMinute("0");
                      setStartSecond("0");
                    }}
                  >
                    Upload Another Video
                  </Button>
                </>
              )}
            </div>


            {frames.length > 0 && (
  <div className="mt-12">
    <h4 className="text-xl font-semibold text-center mb-6">
      Extracted Face Frames
    </h4>

    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
      {frames.map((img, index) => (
        <div
          key={index}
          className="overflow-hidden rounded-xl shadow-md hover:scale-105 transition-transform duration-300"
        >
        <img
          key={index}
          src={`http://localhost:8000${img}`}
          alt="frame"
          className="w-full h-full object-cover"
        />
        </div>
      ))}
    </div>
  </div>
)}
{frameScores.length > 0 && (
  <div className="mt-8">
    <h4 className="text-xl font-semibold text-center mb-4">
      Frame-Level Fake Scores
    </h4>

    <div className="mx-auto grid max-w-4xl grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-5">
      {frameScores.map((item) => (
        <div
          key={item.frame}
          className="rounded-lg border border-border bg-background/60 p-3 text-center text-sm"
        >
          <p className="text-muted-foreground">Frame {item.frame}</p>
          <p className="font-semibold">{(item.fake_score * 100).toFixed(2)}%</p>
        </div>
      ))}
    </div>
  </div>
)}
{heatmaps.length > 0 && (
  <div className="mt-8">
    <h4 className="text-xl font-semibold mb-4 text-center">
      🔥 Grad-CAM Visualization
    </h4>

    <p className="text-center text-muted-foreground mb-4">
      Highlighted regions show where the model focused to make its decision.
    </p>

    <div className="flex flex-wrap gap-4 justify-center">
      {heatmaps.map((img, index) => (
        <div key={index} className="flex flex-col items-center">
          <img
            src={`http://localhost:8000${img}`}
            alt="heatmap"
            className="w-40 rounded shadow border border-primary/30"
          />
          <span className="text-xs mt-1 text-muted-foreground">
            Frame {index + 1}
          </span>
        </div>
      ))}
    </div>
  </div>
)}
{result === "real" && heatmaps.length === 0 && confidence !== null && (
  <p className="mt-8 text-center text-sm text-muted-foreground">
    Grad-CAM heatmaps are generated only when the video is classified as Fake.
  </p>
)}
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

const Brain = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" />
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" />
  </svg>
);
