
use crate::device::Device; #[derive(Clone, Copy, Debug)] pub struct Engine{ pub device:Device }
impl Engine{ pub fn new(device:Device)->Self{Self{device}} pub fn device(&self)->Device{self.device} }
