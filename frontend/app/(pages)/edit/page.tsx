"use client";

import { useState, useEffect } from "react";
import { Box, Typography, TextField, Button, MenuItem, Select, FormControl, InputLabel } from "@mui/material";
import { formatClassName } from "@/core/formatClassName"
import { API_BASE_URL } from "@/shared/constants"

export default function Edit() {
  const [className, setClassName] = useState("");
  const [classes, setClasses] = useState<string[]>([]);
  const [selectedClass, setSelectedClass] = useState<string>("");
  const [message, setMessage] = useState("");
  const [videoFile, setVideoFile] = useState<File | null>(null);

  useEffect(() => {
    const fetchClasses = async () => {
      const res = await fetch(`${API_BASE_URL}/get-classes/`);
      const data = await res.json();
      if (Array.isArray(data.classes)) {
        setClasses(data.classes);
        if (data.classes.length > 0) setSelectedClass(data.classes[0]);
      }
    };
    fetchClasses();
  }, []);

  const handleCreateClass = async () => {
    if (!className.trim()) {
      setMessage("Class name cannot be empty");
      return;
    }

    const classNameFormatted = formatClassName(className);

    try {
      const res = await fetch(`${API_BASE_URL}/create-class/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ className: classNameFormatted }),
      });

      const data = await res.json();
      if (res.ok) {
        setMessage(`Class "${classNameFormatted}" created successfully`);
        setClassName("");
        setClasses((prev) => [...prev, classNameFormatted]);
        setSelectedClass(classNameFormatted);
      } else {
        setMessage(`Error: ${data.message}`);
      }
    } catch (err) {
      console.error(err);
      setMessage("Error creating class");
    }
  };

  const handleVideoUpload = async () => {
    if (!videoFile || !selectedClass) {
      setMessage("Please select a class and video file");
      return;
    }

    const formData = new FormData();
    formData.append("file", videoFile);
    formData.append("label", selectedClass);

    try {
      const res = await fetch(`${API_BASE_URL}/add-video/`, {
        method: "POST", body: formData
      });

      const data = await res.json();
      if (res.ok) {
        setMessage(`Video "${videoFile.name}" uploaded to class "${selectedClass}"`);
        setVideoFile(null);
      } else {
        setMessage(`Error: ${data.message || "Upload failed"}`);
      }
    } catch (err) {
      console.error(err);
      setMessage("Error uploading video");
    }
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", p: 4, backgroundColor: "#121212", minHeight: "100vh", color: "white" }}>
      <Typography variant="h4" mb={2}>Manage Classes</Typography>
      <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
        <TextField
          label="New Class Name"
          value={className}
          onChange={(e) => setClassName(e.target.value)}
          sx={{ input: { color: 'white' }, label: { color: 'rgba(255,255,255,0.7)' }, '& .MuiOutlinedInput-root': { '& fieldset': { borderColor: 'rgba(255,255,255,0.3)' } } }}
        />
        <Button variant="contained" onClick={handleCreateClass}>Create Class</Button>
      </Box>

      <Typography variant="h4" mb={2} mt={4}>Upload Video</Typography>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mb: 2, maxWidth: 400 }}>
        <FormControl
          fullWidth
          sx={{ input: { color: 'white' }, label: { color: 'rgba(255,255,255,0.7)' }, '& .MuiOutlinedInput-root': { '& fieldset': { borderColor: 'rgba(255,255,255,0.3)' } } }}>
          <InputLabel sx={{ color: "white" }}>Select Class</InputLabel>
          <Select value={selectedClass} onChange={(e) => setSelectedClass(e.target.value)} sx={{ color: "white" }}>
            {classes.map((cls) => (
              <MenuItem key={cls} value={cls}>{cls}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <input type="file" accept=".mp4,.avi,.mov" onChange={(e) => e.target.files && setVideoFile(e.target.files[0])} style={{ color: "white" }} />

        <Button variant="contained" onClick={handleVideoUpload}>Upload Video</Button>
      </Box>

      {message && <Typography color="primary">{message}</Typography>}
    </Box>
  );
}
