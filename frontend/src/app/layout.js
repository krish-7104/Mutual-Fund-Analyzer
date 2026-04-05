import "./globals.css";

export const metadata = {
  title: "Mutual Fund Intelligence",
  description: "AI-driven Mutual Fund Analyzer and Financial Advisor",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
