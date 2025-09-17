"use client";

import { useState } from "react";
import { IconButton, Dialog, DialogTitle, DialogContent, Typography, DialogActions, Box } from "@mui/material";
import { CustomButton } from "@/components/CustomButton/CustomButton";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";

export const HelpPopup: React.FC = () => {
  const [open, setOpen] = useState(false);

  const title = "How Classes Work";
  const text = `To name a class, simply enter a name. You can include spaces if you like, for example "Sea and Mountain".
                This class name will automatically be converted to "sea_and_mountain".
                Classes help organize and categorize your videos for easier management and search.`;

  return (
    <>
      <IconButton size="small" onClick={() => setOpen(true)} sx={{ ml: 1 }}>
        <HelpOutlineIcon sx={{ color: "#ffffff" }} />
      </IconButton>

      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        slotProps={{ paper: { sx: { bgcolor: "#1e1f2a", color: "#ededed", borderRadius: 3, p: 2, minWidth: 350 }} }}
      >

        <DialogTitle sx={{ fontWeight: "bold", fontSize: "1.25rem" }}>{title}</DialogTitle>
        <DialogContent>
          <Typography sx={{ whiteSpace: "pre-line", lineHeight: 1.6 }}>{text}</Typography>
        </DialogContent>
        <DialogActions>
          <Box sx={{ width: "100%", display: "flex", justifyContent: "flex-end" }}>
            <CustomButton type="button" label="Close" onClick={() => setOpen(false)} />
          </Box>
        </DialogActions>
      </Dialog>
    </>
  );
};
