import { Typography, Stack, Divider, Link } from "@mui/material";

const techs = [
  { name: "Next.js", url: "https://nextjs.org/", description: "frontend and project structure" },
  { name: "Material UI", url: "https://mui.com/", description: "interface and visual components" },
  { name: "Python & FastAPI", url: "https://fastapi.tiangolo.com/", description: "backend and embedding management" },
  { name: "ChromaDB", url: "https://www.trychroma.com/", description: "video storage and embeddings" },
  { name: "EfficientNet-B4", url: "https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet", description: "video comparison" },
  { name: "Cosine similarity", url: "https://en.wikipedia.org/wiki/Cosine_similarity", description: "calculation method for finding similarities" },
];

export const aboutApplication = (
  <>
    <Typography variant="body1">
      <strong>VideoSearchSimilarity</strong> is an application that allows you to compare an
      input video with a database using embeddings. The goal is to quickly find the most
      visually similar videos in a simple and intuitive way.
    </Typography>
    <Divider sx={{ my: 3, borderColor: "rgba(255, 255, 255, 0.2)" }} />
  </>
);

export const technologiesUsed = (
  <>
    <Typography variant="h5" gutterBottom sx={{ fontWeight: "bold", color: "#ededed" }}>Technologies used</Typography>

    <Stack spacing={1} sx={{ pl: 2 }}>
      {techs.map((tech) => (
        <Link key={tech.name} href={tech.url} target="_blank" underline="hover" color="inherit">
          <Typography>- {tech.name} ({tech.description})</Typography>
        </Link>
      ))}
      <Typography><br />If you want more information on the choice of technologies and see the comparisons and usable alternatives, read the README.md on Github.</Typography>
    </Stack>

    <Divider sx={{ my: 3, borderColor: "rgba(255, 255, 255, 0.2)" }} />
  </>
);

export const aboutAuthor = (
  <>
    <Typography variant="h5" gutterBottom sx={{ fontWeight: "bold", color: "#ededed" }}>About the author</Typography>
    <Typography variant="body1">
      This application was developed by Mercier Charly, a French engineering student, as part of a computer vision internship at the Politehnica University of Timișoara.
      This internship was supervised by lecturer Prof. PHD. Eng. Muguraș Mocofan.<br /><br />
      Find my LinkedIn and GitHub contact below if you have any questions.
    </Typography>
  </>
);
