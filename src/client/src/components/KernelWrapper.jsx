import React, { useEffect } from 'react';
import { useDispatch, useSelector } from "react-redux";
import { makeStyles } from "@material-ui/core/styles";
import { Grid, Box, Typography, List, ListItem, ListItemText } from '@material-ui/core';

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
        
    }, [kernels]);

    return (
        <Box sx={{ p: 1, border: '1px dashed grey' }}>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                GPU Kernels
            </Typography>

            <List>
                {kernels.map((kernel) => (
                    <ListItem key={kernel}>
                        <ListItemText primary={kernel} />
                    </ListItem>
                ))}
            </List>
        </Box>
    )
}

export default KernelWrapper;
