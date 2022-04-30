import * as d3 from "d3";
import React from "react";

import { Typography, Paper } from "@material-ui/core";

function AdjacencyMatrix() {
	var directed = true,
		size = [1, 1],
		nodes = [],
		edges = [],
		edgeWeight = function (d) {
			return d.value;
		},
		nodeID = function (d) {
			return d.id;
		};

	function matrix() {
		const width = size[0],
			height = size[1],
			nodeWidth = width / nodes.length,
			nodeHeight = height / nodes.length,
			constructedMatrix = [],
			matrix = [],
			edgeHash = {},
			xScale = d3.scaleLinear().domain([0, nodes.length]).range([0, width]),
			yScale = d3.scaleLinear().domain([0, nodes.length]).range([0, height]);

		nodes.forEach(function (node, i) {
			node.sortedIndex = i;
		});

		edges.forEach(function (edge) {
			const constructedEdge = {
				source: edge.source,
				target: edge.target,
				weight: edgeWeight(edge),
			};
			if (typeof edge.source == "number") {
				constructedEdge.source = nodes[edge.source];
			}
			if (typeof edge.target == "number") {
				constructedEdge.target = nodes[edge.target];
			}
			const id =
				nodeID(constructedEdge.source) + "-" + nodeID(constructedEdge.target);
	
			edgeHash[id] = constructedEdge;
		});

		nodes.forEach(function (sourceNode, a) {
			nodes.forEach(function (targetNode, b) {
				var grid = {
					id: nodeID(sourceNode) + "-" + nodeID(targetNode),
					source: sourceNode,
					target: targetNode,
					x: xScale(b),
					y: yScale(a),
					weight: 0,
					height: nodeHeight,
					width: nodeWidth,
				};
				var edgeWeight = 0;
				if (edgeHash[grid.id]) {
					edgeWeight = edgeHash[grid.id].weight;
					grid.weight = edgeWeight;
				}
				matrix.push(grid);
				if (directed === false) {
					var mirrorGrid = {
						id: nodeID(sourceNode) + "-" + nodeID(targetNode),
						source: sourceNode,
						target: targetNode,
						x: xScale(a),
						y: yScale(b),
						weight: edgeWeight,
						height: nodeHeight,
						width: nodeWidth,
					};
					matrix.push(mirrorGrid);
				}
			});
		});

		return matrix;
	}

	matrix.directed = function (x) {
		if (!arguments.length) return directed;
		directed = x;
		return matrix;
	};

	matrix.size = function (x) {
		if (!arguments.length) return size;
		size = x;
		return matrix;
	};

	matrix.nodes = function (x) {
		if (!arguments.length) return nodes;
		nodes = x;
		return matrix;
	};

	matrix.links = function (x) {
		if (!arguments.length) return edges;
		edges = x;
		return matrix;
	};

	matrix.edgeWeight = function (x) {
		if (!arguments.length) return edgeWeight;
		if (typeof x === "function") {
			edgeWeight = x;
		} else {
			edgeWeight = function () {
				return x;
			};
		}
		return matrix;
	};

	matrix.nodeID = function (x) {
		if (!arguments.length) return nodeID;
		if (typeof x === "function") {
			nodeID = x;
		}
		return matrix;
	};

	matrix.xAxis = function (calledG) {
		const nameScale = d3
			.scaleBand()
			.domain(nodes.map(nodeID))
			.range([0, size[0]]);

		const xAxis = d3.axisTop(nameScale).tickSize(4);

		calledG
			.append("g")
			.attr("class", "am-xAxis am-axis")
			.call(xAxis)
			.selectAll("text")
			.style("text-anchor", "end")
			.attr("transform", (d) => { `translate(-10,-10) rotate(90)` });
	};

	matrix.yAxis = function (calledG) {
		var nameScale = d3
			.scaleBand()
			.domain(nodes.map(nodeID))
			.range([0, size[1]]);

		const yAxis = d3.axisLeft(nameScale).tickSize(4);

		calledG.append("g").attr("class", "am-yAxis am-axis").call(yAxis);
	};

	return matrix;
}

function Matrix({ name, data }) {
	const id = "matrix-" + name;

	if (Object.keys(data).length !== 0) {
		let { nodes, edges}  = data;

		const adjacencyMatrix = AdjacencyMatrix()
			.size([250, 250])
			.nodeID(function (d) {
				return d.id;
			})
			.edgeWeight((d) => {
				return d.value;
			})
			.nodes(nodes)
			.links(edges)
			.directed(false)
			

		const matrixData = adjacencyMatrix();
		visualize(adjacencyMatrix, matrixData);
	}

	function visualize(wrapper, data) {
		const min = d3.min(data, function (d) {
			return d.weight;
		});
		const max = d3.max(data, function (d) {
			return d.weight;
		});
		const colors = d3.scaleOrdinal(d3.schemeBlues[9]).domain([0, 1]);

		d3.select("#" + id)
			.append("g")
			.attr("transform", "translate(50,50)")
			.attr("id", "adjacencyG")
			.selectAll("rect")
			.data(data)
			.enter()
			.append("rect")
			.attr("width", (d) => d.width)
			.attr("height", (d) => d.height)
			.attr("x", (d) => d.x)
			.attr("y", (d) => d.y)
			.style("stroke", "black")
			.style("stroke-width", "1px")
			.style("fill", function (d) {
				if(max !== min) {
					return colors(d.weight / (max - min));
				} else {
					return colors(0);
				}
			})

		const xAxisLine = d3.select("#" + id)
			.select("#adjacencyG")
			.call(wrapper.xAxis);

		const yAxisLine = d3.select("#" + id)
			.select("#adjacencyG")
			.call(wrapper.yAxis);

		xAxisLine.selectAll("path")
			.style("fill", "none")
			.style("stroke", "black")
			.style("stroke-width", "1px");

		xAxisLine.selectAll("line")
			.style("fill", "none")
			.style("stroke", "black")
			.style("stroke-width", "1px");

		xAxisLine.selectAll("text")
			.style("font-size", "12px")
			.style("font-family", "sans-serif")
			.style("font-weight", "lighter");
	}

	return (
		<Paper>
			<Typography variant="overline" style={{ fontWeight: "bold" }}>
				{name}
			</Typography>
			<svg id={id} width={310} height={310}></svg>
		</Paper>
	);
}

export default Matrix;
