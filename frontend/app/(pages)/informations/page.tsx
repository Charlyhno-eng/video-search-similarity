"use client";

import { Box, Typography, Stack, Paper } from "@mui/material";
import { ContactLinks } from "./_components/Contacts";
import { aboutApplication, technologiesUsed, aboutAuthor } from "./_components/Paragraphs";

export default function Informations() {
  return (
    <Box sx={{ display: "flex", justifyContent: "center", alignItems: "flex-start", minHeight: "100vh", p: 4 }}>
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

        <Box>{aboutApplication}</Box>
        <Box>{technologiesUsed}</Box>
        <Box>{aboutAuthor}</Box>

        <Stack direction="row" spacing={8} sx={{ mt: 4, justifyContent: "center" }}>
          {ContactLinks.map((link) => (
            <Paper
              key={link.title}
              component="a"
              href={link.url}
              target="_blank"
              sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                p: 2,
                textDecoration: "none",
                width: 160,
                bgcolor: "#1e1f2a",
                color: "#ededed",
                borderRadius: 2,
                boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
                "&:hover": { transform: "translateY(-2px)", boxShadow: "0 4px 12px rgba(0,0,0,0.4)" },
                transition: "all 0.2s ease-in-out",
              }}
            >
              {link.icon}
              <Typography sx={{ fontWeight: "bold", mt: 1 }}>{link.title}</Typography>
              <Typography sx={{ fontSize: "0.85rem", textAlign: "center", mt: 0.5 }}> {link.subtitle}</Typography>
            </Paper>
          ))}
        </Stack>
      </Box>
    </Box>
  );
}
