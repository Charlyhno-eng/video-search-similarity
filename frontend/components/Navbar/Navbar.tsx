"use client";

import { AppBar, Toolbar, Box, Link, Divider } from "@mui/material";
import NextLink from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

  const navLinks = [
    { href: "/edit", label: "Edit the database" },
    { href: "/informations", label: "About the application" },
  ];

  return (
    <AppBar position="static" elevation={0} sx={{ backgroundColor: "transparent" }}>
      <Toolbar sx={{ display: "flex", alignItems: "center" }}>

        <Box sx={{ display: "flex", alignItems: "center", gap: 1, flex: 1 }}>
          <Link component={NextLink} href="/" underline="none" sx={{ fontWeight: "bold", letterSpacing: "1px", lineHeight: 1 }}>
            <span style={{ color: "#1976d2", fontSize: "1.8rem" }}>V</span>
            <span style={{ color: "#ffffff", fontSize: "1rem" }}>ideo</span>
            <span style={{ color: "#1976d2", fontSize: "1.8rem" }}>S</span>
            <span style={{ color: "#ffffff", fontSize: "1rem" }}>earch</span>
            <span style={{ color: "#1976d2", fontSize: "1.8rem" }}>S</span>
            <span style={{ color: "#ffffff", fontSize: "1rem" }}>imilarity</span>
          </Link>
        </Box>

        <Box sx={{ flex: 1, display: "flex", justifyContent: "center" }}>
          <Image src="/logo_university.png" alt="Logo Politehnica" width={250} height={50} />
        </Box>

        <Box sx={{ flex: 1, display: "flex", justifyContent: "flex-end", gap: 3 }}>
          {navLinks.map((link) => (
            <Link
              key={link.href}
              component={NextLink}
              href={link.href}
              underline="none"
              color="inherit"
              sx={{
                px: 1.5,
                py: 0.5,
                borderRadius: 1,
                border: pathname === link.href ? "2px solid #1976d2" : "2px solid transparent",
                "&:hover": { borderColor: "#1976d2" },
              }}
            >
              {link.label}
            </Link>
          ))}
        </Box>
      </Toolbar>

      <Divider sx={{ borderColor: "rgba(255, 255, 255, 0.2)" }} />
    </AppBar>
  );
}
