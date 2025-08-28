import React from "react";
import { Box, Card, CardMedia, CardContent, Typography, Grid } from "@mui/material";
import { SimilarVideoType } from "@/app/page"
import { formatFolderName } from "@/utils/formatFolderName"

export function SimilarVideos({ videos }: { videos: SimilarVideoType[] }) {
  if (videos.length === 0) {
    return (
      <Typography variant="body1">No similar videos found yet.</Typography>
    );
  }

  return (
    <Box sx={{ width: "100%" }}>
      <Typography variant="h6" gutterBottom>Most Similar Videos</Typography>

      <Grid container spacing={2}>
        {videos.map((vid, index) => (
          <Grid key={index} size={6}>
            <Card>
              <CardMedia component="img" height="200" image={vid.thumbnail_url} alt={vid.filename} />
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="subtitle1" noWrap>{vid.filename}</Typography>
                  <Typography variant="body2" color="text.secondary">{(vid.similarity * 100).toFixed(2)}%</Typography>
                </Box>

                <Typography variant="body2" color="text.secondary">Video class : {formatFolderName(vid.subfolder)}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
