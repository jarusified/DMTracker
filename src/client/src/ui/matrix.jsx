import * as d3 from "d3";
import React, { useEffect } from 'react';
import { useDispatch, useSelector } from "react-redux";
import { fetchJSON } from "../actions";

import {
    Typography, Paper
} from '@material-ui/core';

function AdjacencyMatrix() {
  var directed = true,
    size = [1,1],
    nodes = [],
    edges = [],
    edgeWeight = function (d) {return 1},
    nodeID = function (d) {return d.id};

  function matrix() {
    const width = size[0],
    height = size[1],
    nodeWidth = width / nodes.length,
    nodeHeight = height / nodes.length,
    constructedMatrix = [],
    matrix = [],
    edgeHash = {},
    xScale = d3.scaleLinear().domain([0,nodes.length]).range([0,width]),
    yScale = d3.scaleLinear().domain([0,nodes.length]).range([0,height]);

    nodes.forEach(function(node, i) {
      node.sortedIndex = i;
    })

    edges.forEach(function(edge) {
      var constructedEdge = {source: edge.source, target: edge.target, weight: edgeWeight(edge)};
      if (typeof edge.source == "number") {
        constructedEdge.source = nodes[edge.source];
      }
      if (typeof edge.target == "number") {
        constructedEdge.target = nodes[edge.target];
      }
      var id = nodeID(constructedEdge.source) + "-" + nodeID(constructedEdge.target);

      if (directed === false && constructedEdge.source.sortedIndex < constructedEdge.target.sortedIndex) {
        id = nodeID(constructedEdge.target) + "-" + nodeID(constructedEdge.source);
      }
      if (!edgeHash[id]) {
        edgeHash[id] = constructedEdge;
      }
      else {
        edgeHash[id].weight = edgeHash[id].weight + constructedEdge.weight;
      }
    });

    console.log("nodes", nodes, nodes.length)

    nodes.forEach(function (sourceNode, a) {
      nodes.forEach(function (targetNode, b) {
        var grid = {id: nodeID(sourceNode) + "-" + nodeID(targetNode), source: sourceNode, target: targetNode, x: xScale(b), y: yScale(a), weight: 0, height: nodeHeight, width: nodeWidth};
        var edgeWeight = 0;
        if (edgeHash[grid.id]) {
          edgeWeight = edgeHash[grid.id].weight;
          grid.weight = edgeWeight;
        };
        if (directed === true || b < a) {
          matrix.push(grid);
          if (directed === false) {
            var mirrorGrid = {id: nodeID(sourceNode) + "-" + nodeID(targetNode), source: sourceNode, target: targetNode, x: xScale(a), y: yScale(b), weight: 0, height: nodeHeight, width: nodeWidth};
            mirrorGrid.weight = edgeWeight;
            matrix.push(mirrorGrid);
          }
        }
      });
    });

    console.log("matrix", matrix, matrix.length)

    return matrix;
  }

  matrix.directed = function(x) {
    if (!arguments.length) return directed;
    directed = x;
    return matrix;
  }

  matrix.size = function(x) {
    if (!arguments.length) return size;
    size = x;
    return matrix;
  }

  matrix.nodes = function(x) {
    if (!arguments.length) return nodes;
    nodes = x;
    return matrix;
  }

  matrix.links = function(x) {
    if (!arguments.length) return edges;
    edges = x;
    return matrix;
  }

  matrix.edgeWeight = function(x) {
    if (!arguments.length) return edgeWeight;
    if (typeof x === "function") {
      edgeWeight = x;
    }
    else {
      edgeWeight = function () {return x};
    }
    return matrix;
  }

  matrix.nodeID = function(x) {
    if (!arguments.length) return nodeID;
    if (typeof x === "function") {
      nodeID = x;
    }
    return matrix;
  }

  matrix.xAxis = function(calledG) {
    const nameScale = d3.scaleOrdinal()
        .domain(nodes.map(nodeID))
        .range([0,size[0]]);

    const xAxis = d3.axisTop(nameScale);

    calledG
        .append("g")
        .attr("class", "am-xAxis am-axis")
        .call(xAxis)
        .selectAll("text")
        .style("text-anchor", "end")
        .attr("transform", "translate(-10,-10) rotate(90)");

  }

  matrix.yAxis = function(calledG) {
    var nameScale = d3.scaleOrdinal()
    .domain(nodes.map(nodeID))
    .range([0,size[1]]);

    const yAxis = d3.axisLeft(nameScale)
    .tickSize(4);

    calledG.append("g")
    .attr("class", "am-yAxis am-axis")
    .call(yAxis);
  }

  return matrix;
}

function Matrix({ name }) {
    const id = "matrix-" + name;
    const dispatch = useDispatch();
    const data = useSelector((store) => store.testJSON);

    useEffect(() => {
        if (data === null) {
            dispatch(fetchJSON("miserables.json"));
        }
        else {
            const adjacencyMatrix = AdjacencyMatrix()
                .size([300,300])
                .nodes(data.nodes)
                .links(data.links)
                .directed(false)
                .nodeID(function (d) {return d.name});  

            const matrixData = adjacencyMatrix();

            visualize(adjacencyMatrix, matrixData);
        }
    }, [data]);

    function visualize(wrapper, data) {
        const colors = d3.scaleOrdinal(d3.schemeCategory10);

        d3.select("#" + id)
            .append("g")
            .attr("transform", "translate(50,50)")
            .attr("id", "adjacencyG")
            .selectAll("rect")
            .data(data)
            .enter()
            .append("rect")
            .attr("width", function (d) {return d.width})
            .attr("height", function (d) {return d.height})
            .attr("x", function (d) {return d.x})
            .attr("y", function (d) {return d.y})
            .style("stroke", "black")
            .style("stroke-width", "0px")
            .style("stroke-opacity", .1)
            .style("fill", function (d) {return colors(d.source.group)})
            .style("fill-opacity", function (d) {return d.weight * .8});

        d3.select("#" + id)
            .select("#adjacencyG")
            .call(wrapper.xAxis);

        d3.select("#" + id)
            .select("#adjacencyG")
            .call(wrapper.yAxis);

    }

    return (
        <Paper>
            <Typography variant="overline" style={{ fontWeight: "bold" }}>
                {name}
			</Typography>
            <svg id={id} width={300} height={300}></svg>
        </Paper>
    )
}

export default Matrix;