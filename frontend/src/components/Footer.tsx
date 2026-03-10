import { Shield, Github, Mail } from "lucide-react";

export const Footer = () => {
  return (
    <footer className="border-t border-border/50 bg-card/30 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Shield className="w-6 h-6 text-primary" />
              <span className="text-xl font-bold">FakeVision</span>
            </div>
            <p className="text-muted-foreground text-sm">
              Advanced deepfake detection using machine learning with visual heatmap explanations.
            </p>
          </div>

          {/* <div>
            <h3 className="font-semibold mb-4">Technologies</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>Artificial Intelligence</li>
              <li>Machine Learning</li>
              <li>Deep Learning</li>
              <li>Computer Vision</li>
              <li>Cybersecurity</li>
            </ul>
          </div> */}
        </div>

        <div className="border-t border-border/50 mt-8 pt-8 text-center text-sm text-muted-foreground">
          <p>&copy; 2026 FakeVision Detector. Built with AI for a safer digital world.</p>
        </div>
      </div>
    </footer>
  );
};
