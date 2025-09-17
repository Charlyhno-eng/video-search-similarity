"use client";

import { Box, FormControl, InputLabel, Select, MenuItem } from "@mui/material";
import { API_BASE_URL, VIDEO_EXTENSIONS } from "@/shared/constants";
import { CustomButton } from "@/components/CustomButton/CustomButton";
import { useState } from "react";
import { Movie } from "@mui/icons-material";

type UploadVideoProps = {
  classes: string[];
  setMessage: React.Dispatch<React.SetStateAction<string>>;
};

export const UploadVideo: React.FC<UploadVideoProps> = ({ classes, setMessage }) => {
  const [selectedClass, setSelectedClass] = useState("");
  const [videoFile, setVideoFile] = useState<File | null>(null);

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
        method: "POST",
        body: formData,
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
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <FormControl fullWidth sx={{ '& .MuiOutlinedInput-root': { '& fieldset': { borderColor: "rgba(255, 255, 255, 0.2)" } } }}>
        <InputLabel sx={{ color: "#ffffff" }}>Select Class</InputLabel>
        <Select value={selectedClass} onChange={(e) => setSelectedClass(e.target.value)} displayEmpty fullWidth sx={{ color: "#ffffff" }}>
          {classes.map((cls) => ( <MenuItem key={cls} value={cls}>{cls}</MenuItem> ))}
        </Select>
      </FormControl>

      <input accept={VIDEO_EXTENSIONS} style={{ display: "none" }} id="upload-video-file" type="file" onChange={(e) => e.target.files && setVideoFile(e.target.files[0])} />
      <label htmlFor="upload-video-file">
        <CustomButton label={videoFile ? `Selected: ${videoFile.name}` : "Choose a video"} component="span" startIcon={<Movie />} />
      </label>

      <CustomButton type="button" label="Upload Video" onClick={handleVideoUpload} />
    </Box>
  );
};
