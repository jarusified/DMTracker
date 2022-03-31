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
	},
}));

function MetricsWrapper() {
	const classes = useStyles();

	const dispatch = useDispatch();
	const selectedKernelMetric = useSelector(
		(store) => store.selected_kernel_metric
	);
	const kernels = useSelector((store) => store.kernels);

	useEffect(() => {
		if (selectedKernelMetric !== "") {
			dispatch(fetchMetrics(selectedKernelMetric));
		}
	}, [selectedKernelMetric]);

	return (
		<Box sx={{ p: 1, border: "1px dashed grey" }}>
			<Typography variant="overline" style={{ fontWeight: "bold" }}>
				Ensemble Performance ({kernels.length} Kernels)
			</Typography>
			<Grid item>
				<Grid item>
					<RuntimeMetrics />
					{/* <TransferMetrics />
					<ProblemSizeMetrics /> */}
				</Grid>
			</Grid>
		</Box>
	);
}

function RuntimeMetrics() {
	const id = "ensemble-runtime-metrics";

	// Get the data from the store.
	const runtimeMetrics = useSelector((store) => store.runtime_metrics);
	const kernelMetrics = useSelector((store) => store.kernel_metrics);
	const transferMetrics = useSelector((store) => store.transfer_metrics);
	const kernels = useSelector((store) => store.kernels);
	const experiments = useSelector((store) => store.experiments);
	const selectedKernelMetric = useSelector((store) => store.selected_kernel_metric);

	// Set dimensions.
	const width = (window.innerHeight/3) * 2;
	const height = window.innerHeight/6;
	const margin = { top: 30, right: 40, bottom: 40, left: 40 };

	// Render the chart when the data changes.
	useEffect(() => {
		if (Object.keys(kernelMetrics).length > 0) {
			const container = d3.select("#" + id);

			// Check if svg element exists inside the container and clear it.
			if (!container.select("svg").empty()) {
				container.select("svg").remove();
			}

			const svg = d3.select("#" + id)
				.append("svg")
				.attr("width", width + margin.left + margin.right)
				.attr("height", height + margin.top + margin.bottom)
				.append("g")
				.attr("transform", `translate(${margin.left}, ${margin.top})`);

			const color = d3.scaleOrdinal().domain(kernels).range(d3.schemeSet2);

			//stack the data
			const stackedData = d3.stack().keys(kernels)(kernelMetrics);

			const x = d3.scaleLinear()
				.domain([0, experiments.length])
				.range([0, width]);

			const xAxis = svg.append("g")
				.attr("transform", `translate(0, ${height})`)
				.call(d3.axisBottom(x).ticks(5));

			// Add X axis label:
			svg.append("text")
				.attr("text-anchor", "end")
				.attr("font-size", "10px")
				.attr("x", width)
				.attr("y", height + 30)
				.text("Experiment");

			// Add Y axis label:
			svg.append("text")
				.attr("font-size", "10px")
				.attr("text-anchor", "end")
				.attr("x", 0)
				.attr("y", -10)
				.text(selectedKernelMetric)
				.attr("text-anchor", "start");

			// Add Y axis
			const y = d3.scaleLinear().domain([0, 20000000]).range([height, 0]);
			svg.append("g")
				.call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".1e")));

			// Add a clipPath: everything out of this area won't be drawn.
			const clip = svg
				.append("defs")
				.append("svg:clipPath")
				.attr("id", "clip")
				.append("svg:rect")
				.attr("width", width)
				.attr("height", height)
				.attr("x", 0)
				.attr("y", 0);

			// Add brushing
			const brush = d3.brushX() // Add the brush feature using the d3.brush function
				.extent([
					[0, 0],
					[width, height],
				]) // initialise the brush area: start at 0,0 and finishes at width,height: it means I select the whole graph area
				.on("end", updateChart); // Each time the brush selection changes, trigger the 'updateChart' function

			// Create the scatter variable: where both the circles and the brush take place
			const areaChart = svg.append("g").attr("clip-path", "url(#clip)");

			// Area generator
			const area = d3.area()
				.x(function (d, i) {
					return x(i);
				})
				.y0(function (d) {
					return y(d[0]);
				})
				.y1(function (d) {
					return y(d[1]);
				});

			// Show the areas
			areaChart.selectAll("mylayers")
				.data(stackedData)
				.join("path")
				.attr("class", function (d) {
					return "myArea " + d.key;
				})
				.style("fill", function (d) {
					return color(d.key);
				})
				.attr("stroke-width", "1px")
				.attr("d", area);

			// Add the brushing
			areaChart.append("g").attr("class", "brush").call(brush);

			// What to do when one group is hovered
			var highlight = function (d) {
				// reduce opacity of all groups
				d3.selectAll(".myArea").style("opacity", 0.1);
				// expect the one that is hovered
				d3.select("." + d).style("opacity", 1);
			};

			// And when it is not hovered anymore
			var noHighlight = function (d) {
				d3.selectAll(".myArea").style("opacity", 1);
			};

			// Add one dot in the legend for each name.
			var size = 10;
			svg.selectAll("myrect")
				.data(kernels)
				.enter()
				.append("rect")
				.attr("x", 400)
				.attr("y", function (d, i) {
					return 10 + i * (size + 5);
				}) // 100 is where the first dot appears. 25 is the distance between dots
				.attr("width", size)
				.attr("height", size)
				.style("fill", function (d) {
					return color(d);
				})
				.on("mouseover", highlight)
				.on("mouseleave", noHighlight);

			// Add one dot in the legend for each name.
			svg.selectAll("mylabels")
				.data(kernels)
				.enter()
				.append("text")
				.attr("x", 400 + size * 1.2)
				.attr("y", function (d, i) {
					return 10 + i * (size + 5) + size / 2;
				}) // 100 is where the first dot appears. 25 is the distance between dots
				.style("fill", function (d) {
					return color(d);
				})
				.text(function (d) {
					return d;
				})
				.attr("text-anchor", "left")
				.style("font-size", "10px")
				.style("alignment-baseline", "middle")
				.on("mouseover", highlight)
				.on("mouseleave", noHighlight);

            let idleTimeout;    
            function idled() { idleTimeout = null; }

			// A function that update the chart for given boundaries
			function updateChart(event, d) {
				let extent = event.selection;

				// If no selection, back to initial coordinate. Otherwise, update X axis domain
				if (!extent) {
					if (!idleTimeout) return (idleTimeout = setTimeout(idled, 350)); // This allows to wait a little bit
					x.domain(
						d3.extent(data, function (d) {
							return d.year;
						})
					);
				} else {
					x.domain([x.invert(extent[0]), x.invert(extent[1])]);
					areaChart.select(".brush").call(brush.move, null); // This remove the grey brush area as soon as the selection has been done
				}

				// Update axis and area position
				xAxis.transition().duration(1000).call(d3.axisBottom(x).ticks(5));
				areaChart.selectAll("path").transition().duration(1000).attr("d", area);
			}
		}
	}, [kernelMetrics, selectedKernelMetric]);

	return (
		<div id={id}></div>
	);
}

// function TransferMetrics() {
// 	const transfer_metrics = useSelector((store) => store.transfer_metrics);

// 	return (
// 		<Fragment>
// 			{transfer_metrics.length > 0 ? (
// 				transfer_metrics.map((metric) => {
// 					<Typography variant="overline" style={{ fontWeight: "bold" }}>
// 						{metric.test} = {metric.mean} {metric.unit}
// 					</Typography>;
// 				})
// 			) : (
// 				<></>
// 			)}
// 		</Fragment>
// 	);
// }

// function ProblemSizeMetrics() {
// 	const problem_size_metrics = useSelector(
// 		(store) => store.problem_size_metrics
// 	);
// 	return (
// 		<Fragment>
// 			{/* {problem_size_metrics != undefined ? (
//                 <Typography variant='overline' style={{ fontWeight: 'bold' }}>
//                     {problem_size_metrics}
//                 </Typography>
//             ) : (<></>)} */}
// 		</Fragment>
// 	);
// }

export default MetricsWrapper;
