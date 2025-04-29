import React, { useState } from "react";

import "../App.css";
import "./Home.css";
import Demo from "./Demo";
import Footer from "../components/Footer";
import Header from "../components/Header";

function Home() {
  //
  // Scroll to demo
  //
  const handleScroll = () => {
    document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div>
      <Header />
      {/* <section id="home">
        <h1 id="title">MA-E-Net</h1>
        <p className="subtitle">
          Semantic Segmentation of Street-level Bike Lanes Using a Residual
          Network Model with Attention Gates
        </p>
        <button className="demo-button" onClick={handleScroll}>Try the Demo</button>
        <Footer />
      </section> */}
      <Demo />
    </div>
  );
}

export default Home;
