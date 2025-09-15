"use client";

import { useState, useEffect } from "react";
import { Container, Typography } from "@mui/material";
import { ManageClasses } from "./_components/ManageClasses";
import { UploadVideo } from "./_components/UploadVideo";
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
    <Container maxWidth="sm" sx={{ display: "flex", flexDirection: "column", justifyContent: "center", minHeight: "100vh" }}>
      <Typography variant="h4" mb={2}>Manage Classes</Typography>
      <ManageClasses setClasses={setClasses} setMessage={setMessage} />

      <Typography variant="h4" mb={2} mt={4}>Upload Video</Typography>
      <UploadVideo classes={classes} setMessage={setMessage} />

      {message && <Typography color="primary">{message}</Typography>}
    </Container>
  );
}
