"use client";

import React from "react";
import { Box, Card, CardMedia, CardContent, Typography, Grid } from "@mui/material";

export type SimilarVideosType = {
  filename: string;
  similarity: number;
  url: string;
};

type SimilarVideosProps = {
  videos: SimilarVideosType[];
};

export function SimilarVideos({ videos }: SimilarVideosProps) {
  if (videos.length === 0) {
    return (
      <Typography variant="body1">No similar videos found yet.</Typography>
    );
  }

  return (
    <Box width="100%">
      <Typography variant="h6" gutterBottom>Most Similar Videos</Typography>

      <Grid container spacing={2}>
        {videos.map((vid, index) => (
          <Grid key={index} size={6}>
            <Card>
              <CardMedia component="video" height="200" src={vid.url} controls />
              <CardContent>
                <Typography variant="body2">
                  <strong>{vid.filename}</strong> - Similarity: {(vid.similarity * 100).toFixed(2)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
