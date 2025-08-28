"use client";

import { VideoSelector } from "../components/VideoSelector/VideoSelector";
import { SimilarVideos } from "../components/SimilarVideos/SimilarVideos";
import { Box } from "@mui/material";
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

  // This callback would be passed to VideoSelector to update similar images
  const handleSimilarVideos = (images: SimilarVideoType[]) => {
    setSimilarVideos(images);
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "row", minHeight: "100vh", p: 2.5 }}>
      <Box sx={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "flex-start" }}>
        <VideoSelector onSimilarVideos={handleSimilarVideos} />
      </Box>

      <Box sx={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "flex-start", pl: 4 }}>
        <SimilarVideos videos={similarVideos} />
      </Box>
    </Box>
  );
}
