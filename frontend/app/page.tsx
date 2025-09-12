"use client";

import { VideoSelector } from "../components/VideoSelector/VideoSelector";
import { SimilarVideos } from "../components/SimilarVideos/SimilarVideos";
import { Box, Divider } from "@mui/material";
import { useState } from "react";

export type SimilarVideoType = {
  filename: string;
  similarity: number;
  url: string;
  thumbnail_url: string;
  subfolder: string;
};

export default function Home() {
  const [similarVideos, setSimilarVideos] = useState<SimilarVideoType[]>([]);

  const handleSimilarVideos = (images: SimilarVideoType[]) => {
    setSimilarVideos(images);
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "row", minHeight: "100vh" }}>
      <Box sx={{ flex: 0.4, display: "flex", justifyContent: "center", alignItems: "flex-start", p: 2.5 }}>
        <VideoSelector onSimilarVideos={handleSimilarVideos} />
      </Box>

      <Divider orientation="vertical" flexItem sx={{ borderColor: "rgba(255, 255, 255, 0.2)" }} />

      <Box sx={{ flex: 0.6, display: "flex", justifyContent: "center", alignItems: "flex-start", p: 2.5 }}>
        <SimilarVideos videos={similarVideos} />
      </Box>
    </Box>
  );
}
