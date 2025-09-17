"use client";

import { useState } from "react";
import { Box, TextField } from "@mui/material";
import { formatClassName } from "@/core/formatClassName";
import { API_BASE_URL } from "@/shared/constants";
import { CustomButton } from "@/components/CustomButton/CustomButton";

type ManageClassesProps = {
  setClasses: React.Dispatch<React.SetStateAction<string[]>>;
  setMessage: React.Dispatch<React.SetStateAction<string>>;
}

export const ManageClasses: React.FC<ManageClassesProps> = ({ setClasses, setMessage }) => {
  const [className, setClassName] = useState("");

  const fetchClasses = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/get-classes/`);
      const data = await res.json();
      if (Array.isArray(data.classes)) {
        setClasses(data.classes.sort((a: string, b: string) => a.localeCompare(b)));
      }
    } catch (err) {
      console.error(err);
      setMessage("Error fetching classes");
    }
  };

  const handleCreateClass = async () => {
    if (!className.trim()) {
      setMessage("Class name cannot be empty");
      return;
    }

    const classNameFormatted = formatClassName(className);

    try {
      const res = await fetch(`${API_BASE_URL}/create-class/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ className: classNameFormatted }),
      });

      const data = await res.json();
      if (res.ok) {
        setMessage(`Class "${classNameFormatted}" created successfully`);
        setClassName("");
        await fetchClasses();
      } else {
        setMessage(`Error: ${data.message}`);
      }
    } catch (err) {
      console.error(err);
      setMessage("Error creating class");
    }
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <TextField
        label="New Class Name"
        value={className}
        onChange={(e) => setClassName(e.target.value.replace(/[^a-zA-Z0-9 ]/g, "")) }
        fullWidth
        size="medium"
        sx={{
          input: { color: 'white' },
          label: { color: 'rgba(255,255,255,0.7)' },
          '& .MuiOutlinedInput-root': { '& fieldset': { borderColor: 'rgba(255,255,255,0.3)' } }
        }}
      />

      <CustomButton type="button" label="Create Class" onClick={handleCreateClass} fullWidth />
    </Box>
  );
};
