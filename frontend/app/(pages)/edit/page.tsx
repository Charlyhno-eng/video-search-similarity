"use client";

import { useState, useEffect } from "react";
import { Container, Typography, Box, Paper } from "@mui/material";
import { ManageClasses } from "./_components/ManageClasses";
import { UploadVideo } from "./_components/UploadVideo";
import { HelpPopup } from "./_components/HelpPopup";
import { API_BASE_URL } from "@/shared/constants";

export default function Edit() {
  const [classes, setClasses] = useState<string[]>([]);
  const [message, setMessage] = useState("");

  const fetchClasses = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/get-classes/`);
      const data = await res.json();
      if (Array.isArray(data.classes)) {
        setClasses(data.classes.sort((a: string, b: string) => a.localeCompare(b)));
      }
    } catch (err) {
      console.error(err);
      setMessage("Error fetching classes");
    }
  };

  useEffect(() => {
    fetchClasses();
  }, []);

  return (
    <Container sx={{ display: "flex", flexDirection: "column", justifyContent: "center", minHeight: "100vh" }}>
      <Box sx={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
        <Paper sx={{ flex: 1, p: 3, border: "1px solid rgba(255, 255, 255, 0.2)", borderRadius: 2, bgcolor: "transparent" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
            <Typography variant="h4" sx={{ color: "#ffffff" }}>Create Classes</Typography>
            <HelpPopup />
          </Box>
          <ManageClasses setClasses={setClasses} setMessage={setMessage} />
        </Paper>

        <Paper sx={{ flex: 1, p: 3, border: "1px solid rgba(255, 255, 255, 0.2)", borderRadius: 2, bgcolor: "transparent" }}>
          <Typography variant="h4" mb={2} sx={{ color: "#ffffff" }}>Upload Video</Typography>
          <UploadVideo classes={classes} setMessage={setMessage} />
        </Paper>
      </Box>

      {message && (<Typography color="primary" sx={{ mt: 3, textAlign: "center" }}>{message}</Typography>)}
    </Container>
  );
}
