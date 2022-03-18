import React, { Fragment, useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { makeStyles } from "@material-ui/core/styles";
import { Grid, Box, Typography } from "@material-ui/core";
import * as d3 from "d3";


import { fetchMetrics } from "../actions";

const useStyles = makeStyles((theme) => ({
    path: {
        stroke: "black",
        strokeWidth: 0.3,
        fill: "none",
    }
}));

function MetricsWrapper() {
	const classes = useStyles();

	const dispatch = useDispatch();
	const selectedExperiment = useSelector((store) => store.selected_experiment);
	const kernels = useSelector((store) => store.kernels);

	useEffect(() => {
		if (selectedExperiment !== "") {
			dispatch(fetchMetrics(selectedExperiment));
		}
	}, [selectedExperiment]);

	return (
		<Box sx={{ p: 1, border: "1px dashed grey" }}>
			<Typography variant="overline" style={{ fontWeight: "bold" }}>
				Ensemble Performance ({kernels.length} Kernels)
			</Typography>
			<Grid item>
				<Grid item>
					<RuntimeMetrics />
					<TransferMetrics />
					<ProblemSizeMetrics />
				</Grid>
			</Grid>
		</Box>
	);
}

function RuntimeMetrics() {
	const id = "ensemble-runtime-metrics";
	const runtimeMetrics = useSelector((store) => store.runtime_metrics);
	const kernelMetrics = useSelector((store) => store.kernel_metrics);
	const transferMetrics = useSelector((store) => store.transfer_metrics);
	const kernels = useSelector((store) => store.kernels);
    const experiments = useSelector((store) => store.experiments);

	const width = window.innerWidth / 3;
	const height = window.innerHeight / 5;
    const margin = {top: 30, right: 0, bottom: 40, left: 50};

	const [selectedMetric, setSelectedMetric] = useState("Global Store Transactions");

	useEffect(() => {
		console.log(runtimeMetrics);
		console.log(transferMetrics);
		console.log(kernelMetrics);
	}, [runtimeMetrics, transferMetrics, kernelMetrics]);

	useEffect(() => {
        if(Object.keys(kernelMetrics).length > 0) {
            const svg = d3.select("#" + id)
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform", `translate(${margin.left}, ${margin.top})`);

            const color = d3.scaleOrdinal()
                .domain(kernels)
                .range(d3.schemeSet2);

            //stack the data
            const stackedData = d3.stack()
                .keys(kernels)(kernelMetrics)

            const x = d3.scaleLinear()
                .domain([0, experiments.length])
                .range([0, width ]);
            const xAxis = svg.append("g")
                .attr("transform", `translate(0, ${height})`)
                .call(d3.axisBottom(x).ticks(5))

            // Add X axis label:
            svg.append("text")
                .attr("text-anchor", "end")
                .attr("font-size", "10px")
                .attr("x", width)
                .attr("y", height + 30 )
                .text("Experiment");

            // Add Y axis label:
            svg.append("text")
                .attr("font-size", "10px")
                .attr("text-anchor", "end")
                .attr("x", 0)
                .attr("y", -10 )
                .text(selectedMetric)
                .attr("text-anchor", "start")

            // Add Y axis
            const y = d3.scaleLinear()
                .domain([0, 20000000])
                .range([ height, 0 ]);
            svg.append("g")
                .call(d3.axisLeft(y).ticks(5))

            // Add a clipPath: everything out of this area won't be drawn.
            const clip = svg.append("defs").append("svg:clipPath")
                .attr("id", "clip")
                .append("svg:rect")
                .attr("width", width )
                .attr("height", height )
                .attr("x", 0)
                .attr("y", 0);

            // Add brushing
            const brush = d3.brushX()                 // Add the brush feature using the d3.brush function
                .extent( [ [0,0], [width,height] ] ) // initialise the brush area: start at 0,0 and finishes at width,height: it means I select the whole graph area
                // .on("end", updateChart) // Each time the brush selection changes, trigger the 'updateChart' function

            // Create the scatter variable: where both the circles and the brush take place
            const areaChart = svg.append('g')
                .attr("clip-path", "url(#clip)")

            // Area generator
            const area = d3.area()
                .x(function(d, i) { console.log(d); return x(i); })
                .y0(function(d) { return y(d[0]); })
                .y1(function(d) { return y(d[1]); })

            // Show the areas
            areaChart
                .selectAll("mylayers")
                .data(stackedData)
                .join("path")
                .attr("class", function(d) { return "myArea " + d.key })
                .style("fill", function(d) { return color(d.key); })
                .attr("stroke-width", "1px")
                .attr("d", area)

            // Add the brushing
            areaChart
                .append("g")
                .attr("class", "brush")
                .call(brush);
        }
	}, [kernelMetrics]);

	return (
		<Fragment>
			<svg id={id} width={width} height={height} pointerEvents="all"></svg>
		</Fragment>
	);
}

function TransferMetrics() {
	const transfer_metrics = useSelector((store) => store.transfer_metrics);

	return (
		<Fragment>
			{transfer_metrics.length > 0 ? (
				transfer_metrics.map((metric) => {
					<Typography variant="overline" style={{ fontWeight: "bold" }}>
						{metric.test} = {metric.mean} {metric.unit}
					</Typography>;
				})
			) : (
				<></>
			)}
		</Fragment>
	);
}

function ProblemSizeMetrics() {
	const problem_size_metrics = useSelector(
		(store) => store.problem_size_metrics
	);
	return (
		<Fragment>
			{/* {problem_size_metrics != undefined ? (
                <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                    {problem_size_metrics}
                </Typography>
            ) : (<></>)} */}
		</Fragment>
	);
}

export default MetricsWrapper;
