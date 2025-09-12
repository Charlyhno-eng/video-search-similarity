import { Button, ButtonProps } from "@mui/material";
import React from "react";

type CustomButtonProps = ButtonProps & {
  label: string;
};

export const CustomButton: React.FC<CustomButtonProps> = ({ label, ...props }) => {
  return (
    <Button
      {...props}
      variant="contained"
      sx={{
        px: 2,
        py: 1,
        backgroundColor: "#115293",
        color: "#ededed",
        "&:hover": { backgroundColor: "#1976d2" },
        "&.Mui-disabled": { backgroundColor: "#115293", color: "#ededed", opacity: 0.7 },
      }}
    >
      {label}
    </Button>
  );
};
