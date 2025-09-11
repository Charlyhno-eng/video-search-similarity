"use client";

import { Box, Typography, Divider, Stack, Button } from "@mui/material";
import GitHubIcon from "@mui/icons-material/GitHub";
import LinkedInIcon from "@mui/icons-material/LinkedIn";
import { CONTACT_LINKS } from "@/shared/constants";

export default function Informations() {
  return (
    <Box sx={{ display: "flex", justifyContent: "center", alignItems: "flex-start", minHeight: "100vh", p: 4, bgcolor: "#14151d", color: "#ededed" }}>
      <Box sx={{ maxWidth: 800, width: "100%" }}>
        <Typography variant="h4" gutterBottom sx={{ fontWeight: "bold" }}>
          About{" "}
          <span style={{ color: "#1976d2", fontSize: "2.4rem" }}>V</span>
          <span style={{ color: "white", fontSize: "1.8rem" }}>ideo</span>
          <span style={{ color: "#1976d2", fontSize: "2.4rem" }}>S</span>
          <span style={{ color: "white", fontSize: "1.8rem" }}>earch</span>
          <span style={{ color: "#1976d2", fontSize: "2.4rem" }}>S</span>
          <span style={{ color: "white", fontSize: "1.8rem" }}>imilarity</span>
        </Typography>

        <Typography variant="body1">
          <strong>VideoSearchSimilarity</strong> est une application permettant de comparer une
          vidéo ou une image d’entrée avec une base de données grâce à des
          <em> embeddings</em>. L’objectif est de retrouver rapidement les vidéos les plus
          similaires visuellement, de manière simple et intuitive.
        </Typography>

        <Divider sx={{ my: 3, bgcolor: "grey.700" }} />

        <Typography variant="h5" gutterBottom sx={{ fontWeight: "bold", color: "#ededed" }}>Technologies utilisées</Typography>
        <Stack spacing={1} sx={{ pl: 2 }}>
          <Typography>- Next.js (frontend et structure du projet)</Typography>
          <Typography>- Material UI (interface et composants visuels)</Typography>
          <Typography>- Python & FastAPI (backend et gestion des embeddings)</Typography>
          <Typography>- Base de données (stockage des vidéos et embeddings)</Typography>
          <Typography>- OpenAI / modèles d’embedding pour la recherche de similarité</Typography>
        </Stack>

        <Divider sx={{ my: 3, bgcolor: "grey.700" }} />

        <Typography variant="h5" gutterBottom sx={{ fontWeight: "bold", color: "#ededed" }}>À propos de l’auteur</Typography>
        <Typography variant="body1">
          Cette application a été réalisée par <strong>Charly Mercier</strong>, passionné par
          l’intelligence artificielle et le développement web.
        </Typography>

        <Stack direction="row" spacing={2} sx={{ mt: 4 }}>
          <Button
            variant="outlined"
            startIcon={<LinkedInIcon />}
            href={CONTACT_LINKS.linkedin}
            target="_blank"
            sx={{ color: "#1976d2", borderColor: "#1976d2", "&:hover": { backgroundColor: "rgba(25,118,210,0.1)" } }}
          >
            LinkedIn
          </Button>
          <Button
            variant="outlined"
            startIcon={<GitHubIcon />}
            href={CONTACT_LINKS.github}
            target="_blank"
            sx={{ color: "#1976d2", borderColor: "#1976d2", "&:hover": { backgroundColor: "rgba(25,118,210,0.1)" } }}
          >
            GitHub
          </Button>
        </Stack>
      </Box>
    </Box>
  );
}
