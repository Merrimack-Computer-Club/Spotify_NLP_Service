import React, { useEffect, useState } from "react";
import { Chart } from "react-google-charts";


/**
 * BarChart
 * @returns 
 */
function TreeChart({data}) {

  const options = {
    minColor: "#f00",
    midColor: "#ddd",
    maxColor: "#0d0",
    headerHeight: 15,
    fontColor: "black",
    showScale: true,
  };

  function definedData() {
    return data != undefined;
  }

  return (
    <div className="tree-chart">
        {definedData() &&
        <Chart
          chartType="TreeMap"
          width="100%"
          height="400px"
          data={data}
          options={options}
      />
      }
    </div>
  );
}

export default TreeChart