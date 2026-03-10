// import { Card, CardContent } from "@/components/ui/card";
// import { Users } from "lucide-react";

// export const Team = () => {
//   const members = [
//     {
//       name: "Alviya",
//       id: "2201220100019",
//       role: "AI/ML Developer",
//     },
//     {
//       name: "Ankita Singh",
//       id: "2201220100028",
//       role: "AI/ML Developer",
//     },
//   ];

//   return (
//     <section className="py-24 px-4 bg-background/50">
//       <div className="container mx-auto max-w-4xl">
//         <div className="text-center mb-16">
//           <div className="flex items-center justify-center gap-2 mb-4">
//             <Users className="w-8 h-8 text-primary" />
//           </div>
//           <h2 className="text-4xl md:text-5xl font-bold mb-4">
//             Our Team
//           </h2>
//           <p className="text-muted-foreground text-lg">
//             The minds behind FakeVision Detector
//           </p>
//         </div>

//         <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
//           {members.map((member, index) => (
//             <Card
//               key={index}
//               className="bg-card shadow-card border-border/50 hover:shadow-glow transition-all duration-300 hover:scale-105"
//             >
//               <CardContent className="p-8 text-center">
//                 <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-gradient-primary flex items-center justify-center text-3xl font-bold text-primary-foreground">
//                   {member.name.charAt(0)}
//                 </div>
//                 <h3 className="text-2xl font-bold mb-2">
//                   {member.name}
//                 </h3>
//                 <p className="text-primary mb-2 font-semibold">
//                   {member.role}
//                 </p>
//                 <p className="text-sm text-muted-foreground">
//                   ID: {member.id}
//                 </p>
//               </CardContent>
//             </Card>
//           ))}
//         </div>

//         <Card className="mt-12 bg-gradient-to-br from-card to-card/50 border-border/50">
//           <CardContent className="p-8">
//             <h3 className="text-xl font-bold mb-4 text-center">Project Technologies</h3>
//             <div className="flex flex-wrap justify-center gap-3">
//               {["Python", "Machine Learning", "CNN", "Grad-CAM", "Computer Vision", "Deep Learning", "PyTorch"].map((tech) => (
//                 <span
//                   key={tech}
//                   className="px-4 py-2 bg-primary/10 text-primary rounded-full text-sm font-medium border border-primary/20"
//                 >
//                   {tech}
//                 </span>
//               ))}
//             </div>
//           </CardContent>
//         </Card>
//       </div>
//     </section>
//   );
// };
