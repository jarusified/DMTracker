import React from 'react';
import {
    Typography, Paper
} from '@material-ui/core';

function Matrix({ name }) {
    return (
        <Paper>
            
            <Typography variant="overline" style={{ fontWeight: "bold" }}>
                {name}
			</Typography>
            <svg width={200} height={250}></svg>
        </Paper>
    )
}

export default Matrix;