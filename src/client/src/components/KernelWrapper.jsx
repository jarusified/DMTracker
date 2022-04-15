import React, { useEffect } from 'react';
import { useDispatch, useSelector } from "react-redux";
import { makeStyles } from "@material-ui/core/styles";
import { Paper, Typography, List, ListItem, ListItemText } from '@material-ui/core';
import { fetchKernels } from "../actions";


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


function KernelWrapper() {
    const dispatch = useDispatch();
    const kernels = useSelector((store) => store.kernels);

    useEffect(() => {
        if(kernels.length === 0) {
            dispatch(fetchKernels())
        }
    }, [kernels]);

    return (
        <Paper>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Kernels view ({kernels.length} Kernels)
            </Typography>

            <List>
                {kernels.map((kernel) => (
                    <ListItem key={kernel}>
                        <ListItemText primary={kernel} />
                    </ListItem>
                ))}
            </List>
        </Paper>
    )
}

export default KernelWrapper;
