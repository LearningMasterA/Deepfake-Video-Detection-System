import { Brain, Thermometer, ShieldCheck } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import videoAnalysisIcon from "@/assets/video-analysis-icon.png";
import heatmapIcon from "@/assets/heatmap-icon.png";
import securityIcon from "@/assets/security-icon.png";

export const Features = () => {
  const features = [
    {
      icon: videoAnalysisIcon,
      title: "Video Analysis",
      description: "Advanced CNN-based deep learning model analyzes video frames to detect manipulation patterns and deepfake artifacts.",
      lucideIcon: Brain,
      gradient: "from-primary/20 to-primary/5",
    },
    {
      icon: heatmapIcon,
      title: "Heatmap Visualization",
      description: "Grad-CAM generates visual heatmaps highlighting suspicious regions, providing transparent and explainable detection results.",
      lucideIcon: Thermometer,
      gradient: "from-warning/20 to-warning/5",
    },
    {
      icon: securityIcon,
      title: "Trusted Security",
      description: "Protect your content with state-of-the-art AI security. Our model is trained on extensive datasets for reliable detection.",
      lucideIcon: ShieldCheck,
      gradient: "from-success/20 to-success/5",
    },
  ];

  return (
    <section className="py-24 px-4">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Powerful Detection Features
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Our advanced AI system combines multiple technologies to provide accurate and transparent deepfake detection
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {features.map((feature, index) => (
            <Card
              key={index}
              className="bg-card shadow-card border-border/50 hover:shadow-glow transition-all duration-300 hover:scale-105 group"
            >
              <CardContent className="p-8">
                <div className={`w-24 h-24 mx-auto mb-6 rounded-2xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center group-hover:scale-110 transition-transform duration-300`}>
                  <img 
                    src={feature.icon} 
                    alt={feature.title} 
                    className="w-16 h-16 object-contain"
                  />
                </div>
                <h3 className="text-2xl font-bold mb-4 text-center">
                  {feature.title}
                </h3>
                <p className="text-muted-foreground text-center leading-relaxed">
                  {feature.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};
