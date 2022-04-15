import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import { Grid, Paper, Typography } from "@material-ui/core";
import { makeStyles } from "@material-ui/core/styles";
import { Timeline } from "vis-timeline";
import { DataSet } from "vis-data";
import "vis-timeline/dist/vis-timeline-graph2d.css";

import { fetchTimeline } from "../actions";

const useStyles = makeStyles((theme) => ({
	timeline: {
		width: window.innerWidth - 20,
		height: 250,
	},
}));

function TimelineWrapper() {
	const classes = useStyles();

    const dispatch = useDispatch();
    const selectedExperiment = useSelector((store) => store.selectedExperiment);
	const timeline = useSelector((store) => store.timeline);

    useEffect(() => {
        if(selectedExperiment !== '') {
            dispatch(fetchTimeline(selectedExperiment));
        }
    }, [selectedExperiment]);

	useEffect(() => {
		const container = document.getElementById("timeline-view");

		const items = new DataSet(timeline);

		// Configuration for the Timeline
		const options = {};

		// Create a Timeline
		const tx = new Timeline(container, items, options);
	}, [timeline]);
	return (
		<Paper>
			<Typography variant="overline" style={{ fontWeight: "bold" }}>
				Execution Timeline
			</Typography>
			<Grid container>
				<Grid item>
					<div id="timeline-view" className={classes.timeline}></div>
				</Grid>
				<Grid item>

				</Grid>
			</Grid>
		</Paper>
	);
}

export default TimelineWrapper;
