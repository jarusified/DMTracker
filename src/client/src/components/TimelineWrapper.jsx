import React, { useEffect } from "react";
import { Grid, Box, Typography } from "@material-ui/core";
import { makeStyles } from "@material-ui/core/styles";
import { Timeline } from "vis-timeline";
import { DataSet } from "vis-data";
import "vis-timeline/dist/vis-timeline-graph2d.css";

const useStyles = makeStyles((theme) => ({
	timeline: {
		width: window.innerWidth,
		height: window.innerHeight / 4,
		border: "1px solid lightgray",
	},
}));

function TimelineWrapper() {
	const classes = useStyles();

	useEffect(() => {
		const container = document.getElementById("timeline-view");

		const items = new DataSet([
			{ id: 1, content: "item 1", start: "2014-04-20" },
			{ id: 2, content: "item 2", start: "2014-04-14" },
			{ id: 3, content: "item 3", start: "2014-04-18" },
			{ id: 4, content: "item 4", start: "2014-04-16", end: "2014-04-19" },
			{ id: 5, content: "item 5", start: "2014-04-25" },
			{ id: 6, content: "item 6", start: "2014-04-27", type: "point" },
		]);

		// Configuration for the Timeline
		const options = {};

		// Create a Timeline
		const timeline = new Timeline(container, items, options);
	}, []);
	return (
		<Box sx={{ p: 1, border: "1px dashed grey" }}>
			<Typography variant="overline" style={{ fontWeight: "bold" }}>
				Execution Timeline
			</Typography>
			<Grid item>
				<div id="timeline-view" className={classes.timeline}></div>
			</Grid>
			<Grid item></Grid>
		</Box>
	);
}

export default TimelineWrapper;
