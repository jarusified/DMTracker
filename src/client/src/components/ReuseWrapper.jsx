import React, { Fragment, useEffect, useState, useRef } from "react";
import * as d3 from "d3";
import { Grid, Paper, Box, Typography } from "@material-ui/core";
import { useSelector } from "react-redux";
import { useResizeObserver } from "beautiful-react-hooks";

import { LOC_LEGEND } from "../helpers/utils";

const margins = {
	top: 0,
	right: 0,
	bottom: 0,
	left: 0,
};

function ReuseWrapper() {
	const cpu = useSelector((store) => store.cpuData);
	const gpu = useSelector((store) => store.gpuData);

	return (
		<Box sx={{ p: 1, border: "1px dashed grey" }}>
			<Typography variant="overline" style={{ fontWeight: "bold" }}>
				Data Reuse analysis
			</Typography>
			<Fragment>
				<Grid container item sm md lg xl >
                    <Grid>
    					<OperationGlyph data={cpu} name={"cpu"} />
                    </Grid>
                    <Grid>
                        <OperationGlyph data={gpu} name={"gpu"} />
                    </Grid>
				</Grid>
			</Fragment>
		</Box>
	);
}

function OperationGlyph({ data, name }) {
	const [size, setSize] = useState({ height: null, width: null });
	const glyphRef = useRef(null);
	const glyphOnResize = useResizeObserver(glyphRef, 200);

	useEffect(() => {
		if (data && size.height && size.width) {
			d3.select(`#operation-${name}-svg`).selectAll("*").remove();
			initGlyph();
		}
	}, [data, size]);

	useEffect(() => {
		if (glyphOnResize) {
			setSize({ height: glyphOnResize.height, width: glyphOnResize.width });
		}
	}, [glyphOnResize]);

	function initGlyph() {
		const labelRadius = d3.min([size.width, size.height]) * 0.45;
		const radius = d3.min([size.width, size.height]) * 0.35;
		const thickness = 30;
		const semiRange = 0.5 * Math.PI;

		console.log(data);

		let chartContainer = d3
			.select(`#operation-${name}-svg`)
			.append("g")
			.attr(
				"transform",
				`translate(${size.width / 2 - 5}, ${size.height / 2 - margins.top})`
			);

		const inputLabel = d3
			.select(`#operation-${name}-svg`)
			.append("text")
			.attr(
				"transform",
				`translate(${margins.left}, ${size.height - margins.bottom})`
			)
			.style("font-size", "0.9rem")
			.text("Kernel Inputs");

		const outputLabel = d3
			.select(`#operation-${name}-svg`)
			.append("text")
			.attr(
				"transform",
				`translate(${size.width / 2 + margins.left + 70}, ${
					size.height - margins.bottom
				})`
			)
			.style("font-size", "0.9rem")
			.text("Kernel Outputs");

		const colorScale = d3
			.scaleOrdinal()
			.domain(LOC_LEGEND.domain)
			.range(LOC_LEGEND.range);

		const getPie = d3
			.pie()
			.value((d) => Math.abs(d.size))
			.sort(null)
			.startAngle(-semiRange)
			.endAngle(semiRange);

		const drawArc = d3
			.arc()
			.outerRadius(radius)
			.innerRadius(radius - thickness);

		//Just for positioning
		const labelArc = d3.arc().outerRadius(labelRadius).innerRadius(labelRadius);

		const inputs = chartContainer
			.append("g")
			.selectAll("path")
			.data(getPie(data.inputs))
			.join("path")
			.attr("d", drawArc)
			.attr("fill", (d) => colorScale(d.data.loc))
			.attr("transform", `rotate(-90)`)
			.attr("stroke", "white")
			.style("stroke-width", 2)
			.style("opacity", 0.8);

		const outputs = chartContainer
			.append("g")
			.selectAll("path")
			.data(getPie(data.outputs))
			.join("path")
			.attr("d", drawArc)
			.attr("fill", (d) => colorScale(d.data.loc))
			.attr("transform", `translate(5, 0) rotate(90)`)
			.attr("stroke", "white")
			.style("stroke-width", 2)
			.style("opacity", 0.8);

		//NOTE: Very quirky way to fix the text position
		const outputLabels = chartContainer
			.append("g")
			.attr(
				"transform",
				`translate(${d3.min([size.width, size.height]) * 0.1}, 0) rotate(90)`
			)
			.selectAll("text")
			.data(getPie(data.outputs))
			.join("text")
			.attr(
				"transform",
				(d) => `translate(${labelArc.centroid(d)}) rotate(-90)`
			)
			.attr("text-anchor", "middle")
			.attr("font-size", "0.7rem")
			.text((d) => `${d.data.size} bytes`);
	}

	return (
			<Paper>
				<Grid item>
					<Typography variant="overline" style={{ fontWeight: "bold" }}>
						{name} Operation - {data ? data.op_id : ""}
					</Typography>
				</Grid>
				<Grid item>
					<div ref={glyphRef} className="viz-container">
						<svg id={`operation-${name}-svg`} width="100%" height="100%"></svg>
					</div>
				</Grid>
			</Paper>
	);
}

export default ReuseWrapper;
