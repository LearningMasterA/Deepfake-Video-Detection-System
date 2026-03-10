import { useState } from "react";
import { Upload, FileVideo, AlertCircle, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { analyzeVideoAPI } from "@/lib/api";

export const UploadSection = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<"real" | "fake" | null>(null);
  const [frames, setFrames] = useState<string[]>([]);
  const [confidence, setConfidence] = useState<number | null>(null);
  const { toast } = useToast();

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
    }
  };

  const analyzeVideo = async () => {
  if (!file) return;

  setIsAnalyzing(true);
  setResult(null);

  try {
    const data = await analyzeVideoAPI(file);

    setResult(data.prediction === "Real" ? "real" : "fake");
    setFrames(data.frames || []);
    setConfidence(data.confidence);

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
  }

  setIsAnalyzing(false);
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
                    <Button
                      onClick={analyzeVideo}
                      className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow"
                    >
                      <Brain className="mr-2 h-4 w-4" />
                      Analyze Video
                    </Button>
                  )}
                  {isAnalyzing && (
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                      <span className="text-muted-foreground">Analyzing...</span>
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
                      setConfidence(null);
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
