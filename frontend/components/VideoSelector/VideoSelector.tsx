"use client";

import React, { useState, useCallback } from "react";
import { Button, Box, Card, CardContent, CardMedia, Typography } from "@mui/material";
import { Movie } from "@mui/icons-material";

export type SimilarVideo = {
  filename: string;
  similarity: number;
  url: string;
};

type BackendResponse = {
  filename: string;
  message: string;
  similar_videos: {
    filename: string;
    similarity: number;
    url: string;
  }[];
};

type VideoSelectorProps = {
  onSimilarVideos: (videos: SimilarVideo[]) => void;
};

export function VideoSelector({ onSimilarVideos }: VideoSelectorProps) {
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [backendResponse, setBackendResponse] = useState<BackendResponse | null>(null);

  const uploadVideo = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:8000/upload-video/", {
      method: "POST", body: formData
    });

    if (!response.ok) throw new Error("Video upload failed");

    return response.json();
  };

  const handleVideoChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const videoURL = URL.createObjectURL(file);
      setSelectedVideo(videoURL);

      try {
        const response: BackendResponse = await uploadVideo(file);
        setBackendResponse(response);

        const similarVideos: SimilarVideo[] = response.similar_videos.map(v => ({
          filename: v.filename,
          similarity: v.similarity,
          url: v.url,
        }));

        onSimilarVideos(similarVideos);

      } catch (error) {
        console.error("Upload failed:", error);
      }
    },
    [onSimilarVideos],
  );

  return (
    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, mt: 4 }}>
      <input
        accept="video/mp4"
        style={{ display: "none" }}
        id="contained-button-file"
        type="file"
        onChange={handleVideoChange}
      />
      <label htmlFor="contained-button-file">
        <Button
          variant="contained"
          component="span"
          startIcon={<Movie />}
          sx={{ bgcolor: "primary.main", "&:hover": { bgcolor: "primary.dark" }, textTransform: "none", px: 3, py: 1 }}
        >
          Select a video
        </Button>
      </label>

      {selectedVideo && (
        <Card sx={{ maxWidth: 800, width: "100%", mt: 2 }}>
          <CardMedia component="video" src={selectedVideo} controls sx={{ maxHeight: 500, objectFit: "contain" }} />
          <CardContent>
            {backendResponse && (
              <Typography variant="body2" color="text.secondary" mt={1}>Backend: {backendResponse.message}</Typography>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
