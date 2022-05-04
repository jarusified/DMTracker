import React, { Fragment, useState, useEffect } from 'react';
import { useDispatch, useSelector } from "react-redux";
import { makeStyles } from "@material-ui/core/styles";
import { Grid, Paper, Typography } from '@material-ui/core';
import DagreGraph from 'dagre-d3-react';
import { fetchCCT } from "../actions";

const useStyles = makeStyles((theme) => ({
	nodes: {
	    fill: "darkgray",
        color: "white",
    },
    path: {
        stroke: "black",
        fill: "black",
        strokeWidth: 1.5,
    }
}));


function CCTWrapper() {
    const classes = useStyles();

    const dispatch = useDispatch();
    const selectedExperiment = useSelector((store) => store.selected_experiment);
	const cct = useSelector((store) => store.cct);
    const [nodes, setNodes] = useState([]);
    const [links, setLinks] = useState([]);

    useEffect(() => {
        if(selectedExperiment !== '') {
            dispatch(fetchCCT(selectedExperiment));
        }
    }, [selectedExperiment]);

    useEffect(() => {
        if(Object.keys(cct).length > 0) {
            setNodes(cct.nodes);
            setLinks(cct.links);
        }        
    }, [cct]);

    return (
        <Paper>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Calling Context Tree
            </Typography>
            <Grid container>
                <DagreGraph
                    nodes={nodes}
                    links={links}
                    config={{
                        rankdir: 'LR',
                        align: 'UL',
                        ranker: 'tight-tree'
                    }}
                    height={window.innerHeight/4}
                    animate={1000}
                    shape='circle'
                    fitBoundaries
                    zoomable
                    onNodeClick={e => console.log(e)}
                    onRelationshipClick={e => console.log(e)}
                />
            </Grid>
        </Paper>
    )
}

function CCT() {
    return (
        <Fragment>
            <svg>
                <g id="container"></g>
            </svg>
        </Fragment>
    )
}


export default CCTWrapper;
