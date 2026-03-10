import { Upload, ScanEye, Activity, CheckCircle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export const HowItWorks = () => {
  const steps = [
    {
      icon: Upload,
      title: "Upload Video",
      description: "Upload your video file through our secure interface. We support all major video formats.",
      step: "01",
    },
    {
      icon: ScanEye,
      title: "Frame Extraction",
      description: "Our system extracts and preprocesses video frames for analysis using advanced computer vision techniques.",
      step: "02",
    },
    {
      icon: Activity,
      title: "CNN Analysis",
      description: "A trained Convolutional Neural Network analyzes frames for deepfake patterns and manipulation artifacts.",
      step: "03",
    },
    {
      icon: CheckCircle,
      title: "Get Results",
      description: "Receive classification results with Grad-CAM heatmaps showing suspicious regions for full transparency.",
      step: "04",
    },
  ];

  return (
    <section id="how-it-works" className="py-24 px-4">
      {/* <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            How It Works
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Our detection process combines cutting-edge AI technology with explainable visualization
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
          {steps.map((step, index) => (
            <Card
              key={index}
              className="bg-card shadow-card border-border/50 relative overflow-hidden group hover:shadow-glow transition-all duration-300"
            >
              <div className="absolute top-4 right-4 text-6xl font-bold text-primary/10 group-hover:text-primary/20 transition-colors">
                {step.step}
              </div>
              <CardContent className="p-6 relative">
                <div className="w-16 h-16 mb-6 rounded-xl bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                  <step.icon className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-xl font-bold mb-3">
                  {step.title}
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  {step.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-16 max-w-4xl mx-auto">
          <Card className="bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20 shadow-glow">
            <CardContent className="p-8">
              <h3 className="text-2xl font-bold mb-4 text-center">
                Grad-CAM Visualization
              </h3>
              <p className="text-muted-foreground text-center leading-relaxed mb-6">
                Gradient-weighted Class Activation Mapping (Grad-CAM) generates visual heatmaps that highlight 
                the most influential regions in each frame. This provides transparency and helps users understand 
                exactly which areas contributed to the detection decision.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div className="bg-card/50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2 text-primary">Explainable AI</h4>
                  <p className="text-muted-foreground">
                    Visual evidence of manipulation patterns increases trust and understanding
                  </p>
                </div>
                <div className="bg-card/50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2 text-primary">Precise Detection</h4>
                  <p className="text-muted-foreground">
                    Heatmaps pinpoint exact locations of suspicious artifacts in video frames
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div> */}
    </section>
  );
};
