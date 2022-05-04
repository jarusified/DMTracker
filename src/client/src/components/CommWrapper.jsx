import React, { useEffect, useState } from "react";
import { makeStyles } from "@material-ui/core/styles";
import { Grid, Typography, Paper } from "@material-ui/core";
import { useDispatch, useSelector } from "react-redux";
import {
	FormHelperText,
	FormControl,
	InputLabel,
	Select,
	MenuItem,
} from "@mui/material";

import Matrix from "../ui/matrix";
import { fetchComm } from "../actions";

const useStyles = makeStyles((theme) => ({
	rowContainer: {
		display: "grid",
		gridAutoFlow: "column",
		alignItems: "center",
	},
    formControl: {
		padding: 20,
		justifyContent: "flex-end",
		textColor: "white",
		minWidth: 120
	},
}));

function CommWrapper() {
	const classes = useStyles();
	const dispatch = useDispatch();
	const comm_matrix = useSelector((store) => store.comm_matrix);
	const selectedExperiment = useSelector((store) => store.selected_experiment);
	const [h2d, set_h2d] = useState({});
	const [p2p, set_p2p] = useState({});
	const [zc, set_zc] = useState({});
	const modes = ["unified", "explicit"];
	const [mode, set_mode] = useState(modes[0]);
	const metrics = ["bytes", "times"];
	const [metric, set_metric] = useState(metrics[0]);

	useEffect(() => {
		if (selectedExperiment !== "") {
			dispatch(fetchComm(selectedExperiment));
		}
	}, [selectedExperiment]);

	useEffect(() => {
		if (Object.keys(comm_matrix).length !== 0) {
			set_h2d(comm_matrix.H2D[mode][metric]);
			set_p2p(comm_matrix.P2P[mode][metric]);
			set_zc(comm_matrix.ZC[metric]);
		}
	}, [comm_matrix, mode, metric]);

	return (
		<Paper>
			<Grid container spacing={2}>
				<Grid item xs={6}>
					<Typography variant="overline" style={{ fontWeight: "bold" }}>
						Data movement Matrix
					</Typography>
				</Grid>
				<Grid item xs={3}>
					{metrics.length > 0 ? (
						<FormControl className={classes.formControl} size="small">
							<InputLabel id="dataset-label">Metrics</InputLabel>
							<Select
								labelId="dataset-label"
								id="dataset-select"
								value={metric}
								onChange={(e) => {
								    set_metric(e.target.value);
								}}
							>
								{metrics.map((cc) => (
									<MenuItem key={cc} value={cc}>
										{cc}
									</MenuItem>
								))}
							</Select>
						</FormControl>
					) : (
						<></>
					)}
				</Grid>
                <Grid item xs={3}>
                    {modes.length > 0 ? (
                        <FormControl className={classes.formControl} size="small">
                            <InputLabel id="dataset-label">Modes</InputLabel>
                            <Select
                                labelId="dataset-label"
                                id="dataset-select"
                                value={mode}
                                onChange={(e) => {
                                   set_mode(e.target.value);
                                }}
                            >
                                {modes.map((cc) => (
                                    <MenuItem key={cc} value={cc}>
                                        {cc}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    ) : (
                        <></>
                    )}
                </Grid>
			</Grid>

			<Grid container spacing={2}>
				<Grid item xs={4}>
					<Matrix name={"Host-Device"} data={h2d} mode={mode} metric={metric} />
				</Grid>
				<Grid item xs={4}>
					<Matrix name={"Peer-Peer"} data={p2p} mode={mode} metric={metric} />
				</Grid>
				<Grid item xs={4}>
					<Matrix name={"Zero-copy"} data={zc} mode={mode} metric={metric} />
				</Grid>
			</Grid>
		</Paper>
	);
}

export default CommWrapper;
