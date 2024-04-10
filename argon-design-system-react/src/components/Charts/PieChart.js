import React, { useEffect, useState } from "react";
import { Chart } from "react-google-charts";


/**
 * BarChart
 * @returns 
 */
function PieChart({data}) {

  const options = {
    title: "Emotional Analysis Breakdown",
    is3D: true,
  };

  function definedData() {
    return data != undefined;
  }

  return (
    <div className="pie-chart">
        {definedData() &&
        <Chart
          chartType="PieChart"
          data={data}
          options={options}
          width={"100%"}
          height={"400px"}
      />
      }
    </div>
  );
}

export default PieChart