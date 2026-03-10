import { Upload, Shield, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import heroBanner from "@/assets/hero-banner.jpg";

export const Hero = () => {
  const scrollToUpload = () => {
    document.getElementById("upload-section")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <div 
        className="absolute inset-0 z-0"
        style={{
          backgroundImage: `url(${heroBanner})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          opacity: 0.3,
        }}
      />
      <div className="absolute inset-0 bg-gradient-hero z-0" />
      
      <div className="container relative z-10 px-4 py-20">
        <div className="max-w-4xl mx-auto text-center">
          <div className="mb-6 flex items-center justify-center gap-2">
            <Shield className="w-8 h-8 text-primary animate-pulse" />
            <span className="text-primary font-semibold tracking-wide uppercase text-sm">
              AI-Powered Security
            </span>
          </div>

          <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-primary bg-clip-text text-transparent">
            Deepfake Video Detection with Visual Explanation
          </h1>

          <p className="text-xl md:text-2xl text-muted-foreground mb-8 leading-relaxed">
            Advanced deepfake detection using machine learning with visual heatmap explanations
          </p>

          <p className="text-lg text-foreground/80 mb-12 max-w-2xl mx-auto">
            Protect yourself from AI-generated fake videos. Our CNN-based system analyzes video content 
            and provides transparent, explainable results with Grad-CAM visualization.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              size="lg"
              onClick={scrollToUpload}
              className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow transition-all duration-300 hover:scale-105"
            >
              <Upload className="mr-2 h-5 w-5" />
              Analyze Video
            </Button>
            {/* <Button
              size="lg"
              variant="outline"
              onClick={() => document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" })}
              className="border-primary/50 hover:bg-primary/10 hover:border-primary"
            >
              <Eye className="mr-2 h-5 w-5" />
              How It Works
            </Button> */}
          </div>

          <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
            <div className="p-6 bg-card/50 backdrop-blur-sm rounded-lg border border-border/50">
              <div className="text-3xl font-bold text-primary mb-2">99.2%</div>
              <div className="text-sm text-muted-foreground">Detection Accuracy</div>
            </div>
            <div className="p-6 bg-card/50 backdrop-blur-sm rounded-lg border border-border/50">
              <div className="text-3xl font-bold text-primary mb-2">&lt;5s</div>
              <div className="text-sm text-muted-foreground">Processing Time</div>
            </div>
            <div className="p-6 bg-card/50 backdrop-blur-sm rounded-lg border border-border/50">
              <div className="text-3xl font-bold text-primary mb-2">CNN + Grad-CAM</div>
              <div className="text-sm text-muted-foreground">Advanced AI Models</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
