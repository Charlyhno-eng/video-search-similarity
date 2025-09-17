"use client";

import { useState, useEffect, useCallback } from "react";
import { Box, FormControl, InputLabel, Select, MenuItem, IconButton, Typography, Paper } from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import { API_BASE_URL } from "@/shared/constants";

type Video = {
  filename: string;
  video_path: string;
  thumbnail_path: string;
};

type ManageVideosProps = {
  classes: string[];
  setMessage: React.Dispatch<React.SetStateAction<string>>;
};

export const ManageVideos: React.FC<ManageVideosProps> = ({ classes, setMessage }) => {
  const [selectedClass, setSelectedClass] = useState("");
  const [videos, setVideos] = useState<Video[]>([]);

  // Fetch videos for a given class
  const fetchVideos = useCallback(async (cls: string) => {
    if (!cls) return;
    try {
      const res = await fetch(`${API_BASE_URL}/list-videos/?class_name=${cls}`);
      const data = await res.json();
      setVideos(data.videos || []);
    } catch (err) {
      console.error(err);
      setMessage("Error fetching videos");
    }
  }, [setMessage]);

  // Fetch videos when selectedClass changes
  useEffect(() => {
    if (selectedClass) fetchVideos(selectedClass);
  }, [selectedClass, fetchVideos]);

  const handleDelete = async (filename: string) => {
    if (!selectedClass) return;
    try {
      const formData = new FormData();
      formData.append("class_name", selectedClass);
      formData.append("filename", filename);

      const res = await fetch(`${API_BASE_URL}/delete-video/`, {
        method: "DELETE", body: formData
      });

      const data = await res.json();
      if (res.ok) {
        setMessage(data.message);
        setVideos(videos.filter(v => v.filename !== filename));
      } else {
        setMessage(`Error deleting video: ${data.message || "Failed"}`);
      }
    } catch (err) {
      console.error(err);
      setMessage("Error deleting video");
    }
  };

  return (
    <Paper sx={{ p: 3, border: "1px solid rgba(255, 255, 255, 0.2)", borderRadius: 2, bgcolor: "transparent", width: "95%" }}>
      <FormControl fullWidth sx={{ mb: 2,  '& .MuiOutlinedInput-root': { '& fieldset': { borderColor: "rgba(255, 255, 255, 0.2)" } } }}>
        <InputLabel sx={{ color: "#ffffff" }}>Select Class</InputLabel>
        <Select value={selectedClass} onChange={(e) => setSelectedClass(e.target.value)} sx={{ color: "#ffffff" }}>
          {classes.map(cls => (<MenuItem key={cls} value={cls}>{cls}</MenuItem>))}
        </Select>
      </FormControl>

      <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
        {videos.length === 0 && selectedClass && ( <Typography sx={{ color: "#ffffff" }}>No videos in this class.</Typography> )}
        {videos.map(v => (
          <Box key={v.filename} sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", p: 1, border: "1px solid rgba(255,255,255,0.1)", borderRadius: 1 }}>
            <Typography sx={{ color: "#ffffff", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v.filename}</Typography>
            <IconButton onClick={() => handleDelete(v.filename)} size="small" sx={{ color: "red" }}>
              <DeleteIcon />
            </IconButton>
          </Box>
        ))}
      </Box>
    </Paper>
  );
};
