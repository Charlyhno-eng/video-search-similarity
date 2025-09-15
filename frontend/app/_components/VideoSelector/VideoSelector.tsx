"use client";

import React, { useState, useCallback } from "react";
import { Box, Card, CardContent, CardMedia, Typography, CircularProgress } from "@mui/material";
import { CustomButton } from "@/components/CustomButton/CustomButton";
import { Movie } from "@mui/icons-material";
import { SimilarVideoType } from "@/app/page";
import { API_BASE_URL, VIDEO_EXTENSIONS } from "@/shared/constants"

type BackendResponse = {
  filename: string;
  message: string;
  thumbnail_url: string;
  similar_videos: SimilarVideoType[];
};

export function VideoSelector({ onSimilarVideos }: { onSimilarVideos: (videos: SimilarVideoType[]) => void }) {
  const [backendResponse, setBackendResponse] = useState<BackendResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const uploadVideo = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/upload-video/`, {
      method: "POST", body: formData
    });

    if (!response.ok) throw new Error("Video upload failed");
    return response.json();
  };

  const handleVideoChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      setLoading(true);
      try {
        const response: BackendResponse = await uploadVideo(file);
        setBackendResponse(response);

        const similarVideos: SimilarVideoType[] =
          response.similar_videos.map((v) => ({
            filename: v.filename,
            similarity: v.similarity,
            url: v.url,
            thumbnail_url: v.thumbnail_url,
            subfolder: v.subfolder,
          }));

        onSimilarVideos(similarVideos);
      } catch (error) {
        console.error("Upload failed:", error);
      } finally {
        setLoading(false);
      }
    }, [onSimilarVideos]
  );

  return (
    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", mt: 10, width: "100%" }}>
      <input accept={VIDEO_EXTENSIONS} style={{ display: "none" }} id="contained-button-file" type="file" onChange={handleVideoChange} />
      <label htmlFor="contained-button-file">
        <CustomButton
          label={loading ? "Uploading..." : "Select a video"}
          component="span"
          startIcon={<Movie />}
          disabled={loading}
        />
      </label>

      {loading && (
        <Box sx={{ mt: 4 }}><CircularProgress size="3rem" /></Box>
      )}

      {!loading &&
        backendResponse &&
        backendResponse.similar_videos.length > 0 && (
          <Card sx={{ width: "100%", mt: 4 }}>
            <CardMedia component="img" height="350" image={backendResponse.similar_videos[0].thumbnail_url} alt={backendResponse.similar_videos[0].filename} />
            <CardContent>
              <Typography variant="subtitle1" noWrap>{backendResponse.filename}</Typography>
              <Typography variant="body2" color="text.secondary" mt={1}>Backend: {backendResponse.message}</Typography>
            </CardContent>
          </Card>
        )}
    </Box>
  );
}
